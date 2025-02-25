import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from utils.logger import setup_logger

logger = setup_logger("price_prediction")


class PricePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=24):
        super().__init__()
        self.input_size = input_size  # 特征数（如价格、情绪等）
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size  # 预测未来24小时价格
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.model_path = "models/price_predictor.pth"
        self.scaler = MinMaxScaler()  # 用于数据归一化

    def forward(self, x):
        """前向传播"""
        out, _ = self.lstm(x)  # out: [batch_size, timesteps, hidden_size]
        out = self.fc(out[:, -1, :])  # 取最后一个时间步，预测未来24小时
        return out

    def preprocess_data(self, data):
        """预处理输入数据"""
        try:
            # 假设data是[time_steps, features]的二维数组
            data_array = np.array(data)
            if data_array.ndim != 2 or data_array.shape[1] != self.input_size:
                raise ValueError(f"输入数据维度错误，期望[time_steps, {self.input_size}]，实际{data_array.shape}")

            # 只对价格列（假设第0列）进行归一化
            price_data = data_array[:, 0].reshape(-1, 1)
            normalized_price = self.scaler.transform(price_data)
            data_array[:, 0] = normalized_price.flatten()
            return torch.tensor(data_array, dtype=torch.float32)
        except Exception as e:
            logger.error(f"数据预处理错误: {e}")
            return None

    def inverse_transform(self, prediction):
        """反归一化预测结果"""
        try:
            prediction_array = prediction.detach().cpu().numpy()
            return self.scaler.inverse_transform(prediction_array.reshape(-1, 1)).flatten()
        except Exception as e:
            logger.error(f"反归一化错误: {e}")
            return prediction

    def predict(self, data):
        """预测未来24小时价格"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("未找到训练好的模型，请先训练")

            self.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.eval()
            processed_data = self.preprocess_data(data)
            if processed_data is None:
                return None

            with torch.no_grad():
                input_tensor = processed_data.unsqueeze(0)  # [1, time_steps, features]
                prediction = self.forward(input_tensor)  # [1, 24]
                predicted_prices = self.inverse_transform(prediction)
                logger.info(f"预测未来24小时价格: {predicted_prices.tolist()}")
                return predicted_prices.tolist()
        except Exception as e:
            logger.error(f"价格预测错误: {e}")
            return None

    def train_model(self, X, y, time_steps=10, epochs=100, batch_size=32, validation_split=0.2):
        """训练价格预测模型"""
        try:
            # 数据预处理
            X_array = np.array(X)  # [samples, features]
            y_array = np.array(y)  # [samples, 24]

            # 拟合scaler（仅对价格列）
            self.scaler.fit(X_array[:, 0].reshape(-1, 1))

            # 转换为时间序列格式
            X_seq, y_seq = [], []
            for i in range(len(X_array) - time_steps):
                X_seq.append(X_array[i:i + time_steps])
                y_seq.append(y_array[i + time_steps - 1])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)

            # 分割训练和验证集
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

            # 转换为Tensor
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

            # 数据加载器
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # 训练设置
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # 训练循环
            for epoch in range(epochs):
                self.train()
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    output = self.forward(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # 验证
                self.eval()
                with torch.no_grad():
                    val_output = self.forward(X_val.to(device))
                    val_loss = criterion(val_output, y_val.to(device))

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")

            torch.save(self.state_dict(), self.model_path)
            logger.info("价格预测模型训练完成并保存")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_X, new_y, time_steps=10, epochs=10):
        """在线更新模型"""
        try:
            self.load_state_dict(torch.load(self.model_path))
            X_seq, y_seq = [], []
            new_X_array = np.array(new_X)
            new_y_array = np.array(new_y)
            for i in range(len(new_X_array) - time_steps):
                X_seq.append(new_X_array[i:i + time_steps])
                y_seq.append(new_y_array[i + time_steps - 1])

            X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
            y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)

            optimizer = optim.Adam(self.parameters(), lr=0.0001)  # 较低的学习率用于微调
            criterion = nn.MSELoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)

            self.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self.forward(X_tensor.to(device))
                loss = criterion(output, y_tensor.to(device))
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    logger.info(f"Update Epoch {epoch}, Loss: {loss.item():.4f}")

            torch.save(self.state_dict(), self.model_path)
            logger.info("模型在线更新完成")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")


if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    prices = np.random.normal(150, 10, 1000)
    X = [[p, 0.5, 0.2, 1000, 200] for p in prices]  # [price, sentiment, holder_growth, liquidity, volume]
    y = [prices[i:i + 24] for i in range(len(prices) - 24)]  # 未来24小时价格

    model = PricePredictor()
    model.train_model(X, y[:len(X) - 24], time_steps=10, epochs=50)

    # 预测
    sample_data = X[-10:]  # 最近10个时间步
    prediction = model.predict(sample_data)
    print(f"预测未来24小时价格: {prediction}")