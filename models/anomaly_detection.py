import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import joblib
import os
import yaml
from utils.db import Database
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger("anomaly_detection")


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetector:
    def __init__(self, if_model_path="models/if_anomaly_detector.pkl", ae_model_path="models/ae_anomaly_detector.pth"):
        self.if_model_path = if_model_path
        self.ae_model_path = ae_model_path

        self.if_model = IsolationForest(contamination=0.1, random_state=42)
        if os.path.exists(if_model_path):
            try:
                self.if_model = joblib.load(if_model_path)
                logger.info("加载预训练孤立森林模型")
            except Exception as e:
                logger.error(f"加载孤立森林模型失败: {e}")

        self.input_dim = 5
        self.ae_model = Autoencoder(self.input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae_model.to(self.device)
        if os.path.exists(ae_model_path):
            try:
                self.ae_model.load_state_dict(torch.load(ae_model_path, map_location=self.device))
                logger.info("加载预训练自编码器模型")
            except Exception as e:
                logger.error(f"加载自编码器模型失败: {e}")
        self.ae_model.eval()

        config = yaml.safe_load(open("data_pipeline/config.yaml", "r"))
        self.db = Database(config["database"])
        self.blacklist = self.db.get_blacklist()
        self.ae_threshold = 0.1

    def update_blacklist(self, address):
        self.db.update_blacklist(address)
        self.blacklist.add(address)
        logger.info(f"添加地址 {address} 到黑名单")

    @staticmethod
    def preprocess_data(data):
        """预处理数据，转换为DataFrame并补齐缺失列"""
        try:
            if isinstance(data, dict):
                data = [data]
            df = pd.DataFrame(data)
            required_cols = ["volume", "price_change", "liquidity_change", "tx_count", "address"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0 if col != "address" else ""
            return df
        except Exception as e:
            logger.error(f"数据预处理错误: {e}")
            return None

    def detect(self, data, batch_size=32):
        try:
            df = self.preprocess_data(data)  # 调用静态方法
            if df is None:
                return [False] * len(data) if isinstance(data, list) else False

            is_blacklisted = df["address"].isin(self.blacklist).tolist()
            if any(is_blacklisted):
                logger.warning(f"检测到 {sum(is_blacklisted)} 个黑名单地址，标记为异常")

            if_features = df.drop(columns=["address"])
            if_predictions = self.if_model.predict(if_features)
            if_anomalies = [pred == -1 for pred in if_predictions]

            ae_features = torch.tensor(if_features.values, dtype=torch.float32).to(self.device)
            ae_anomalies = []
            with torch.no_grad():
                for i in range(0, len(ae_features), batch_size):
                    batch = ae_features[i:i + batch_size]
                    reconstructed = self.ae_model(batch)
                    mse = torch.mean((batch - reconstructed) ** 2, dim=1)
                    ae_anomalies.extend(mse > self.ae_threshold)

            final_anomalies = [if_a or ae_a or bl for if_a, ae_a, bl in zip(if_anomalies, ae_anomalies, is_blacklisted)]
            logger.info(f"检测 {len(df)} 条数据，异常数: {sum(final_anomalies)}")
            return final_anomalies if len(data) > 1 else final_anomalies[0]
        except Exception as e:
            logger.error(f"异常检测错误: {e}")
            return [False] * len(data) if isinstance(data, list) else False

    def train_model(self, data, ae_epochs=50):
        try:
            df = self.preprocess_data(data)  # 调用静态方法
            if df is None:
                return

            if_features = df.drop(columns=["address"])
            self.if_model.fit(if_features)
            joblib.dump(self.if_model, self.if_model_path)
            logger.info("孤立森林模型训练完成并保存")

            ae_features = torch.tensor(if_features.values, dtype=torch.float32).to(self.device)
            optimizer = torch.optim.Adam(self.ae_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            self.ae_model.train()
            for epoch in range(ae_epochs):
                optimizer.zero_grad()
                reconstructed = self.ae_model(ae_features)
                loss = criterion(reconstructed, ae_features)
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    logger.info(f"AE Epoch {epoch}, Loss: {loss.item():.4f}")

            torch.save(self.ae_model.state_dict(), self.ae_model_path)
            logger.info("自编码器模型训练完成并保存")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_data):
        try:
            df = self.preprocess_data(new_data)  # 调用静态方法
            if df is None:
                return

            if_features = df.drop(columns=["address"])
            self.if_model.fit(if_features)
            joblib.dump(self.if_model, self.if_model_path)
            logger.info("孤立森林模型在线更新完成")

            ae_features = torch.tensor(if_features.values, dtype=torch.float32).to(self.device)
            optimizer = torch.optim.Adam(self.ae_model.parameters(), lr=0.0001)
            criterion = nn.MSELoss()

            self.ae_model.train()
            optimizer.zero_grad()
            reconstructed = self.ae_model(ae_features)
            loss = criterion(reconstructed, ae_features)
            loss.backward()
            optimizer.step()
            torch.save(self.ae_model.state_dict(), self.ae_model_path)
            logger.info(f"自编码器在线更新完成，Loss: {loss.item():.4f}")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")


if __name__ == "__main__":
    detector = AnomalyDetector()
    train_data = [
        {"volume": 1000, "price_change": 0.05, "liquidity_change": 100, "tx_count": 50, "address": "addr1"},
        {"volume": 5000, "price_change": 0.5, "liquidity_change": -2000, "tx_count": 10, "address": "addr2"}
    ]
    detector.train_model(train_data)
    test_data = [
        {"volume": 6000, "price_change": 0.6, "liquidity_change": -3000, "tx_count": 5, "address": "addr3"},
        {"volume": 800, "price_change": 0.02, "liquidity_change": 50, "tx_count": 40, "address": "addr1"}
    ]
    anomalies = detector.detect(test_data)
    print(f"异常检测结果: {anomalies}")
    detector.update_blacklist("addr3")
    anomalies_after_blacklist = detector.detect(test_data)
    print(f"黑名单后异常检测结果: {anomalies_after_blacklist}")