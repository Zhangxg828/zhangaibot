import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils.logger import setup_logger
import os

logger = setup_logger("price_prediction")


class PricePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=24):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.model_path = "models/price_predictor.pth"
        self.scaler = MinMaxScaler()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def preprocess_data(self, _prep_data):
        """预处理输入数据"""
        try:
            data_array = np.array(_prep_data)
            if data_array.ndim != 2 or data_array.shape[1] != self.input_size:
                raise ValueError(f"输入数据维度错误，期望[time_steps, {self.input_size}]，实际{data_array.shape}")
            price_data = data_array[:, 0].reshape(-1, 1)
            normalized_price = self.scaler.transform(price_data)
            data_array[:, 0] = normalized_price.flatten()
            return torch.tensor(data_array, dtype=torch.float32)
        except Exception as e:
            logger.error(f"数据预处理错误: {e}")
            return None

    def inverse_transform(self, pred):
        try:
            pred_array = pred.detach().cpu().numpy()
            return self.scaler.inverse_transform(pred_array.reshape(-1, 1)).flatten()
        except Exception as e:
            logger.error(f"反归一化错误: {e}")
            return pred

    def predict(self, _predict_data):
        """预测未来24小时价格"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("未找到训练好的模型，请先训练")
            self.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.eval()
            processed_data = self.preprocess_data(_predict_data)
            if processed_data is None:
                return None
            with torch.no_grad():
                input_tensor = processed_data.unsqueeze(0)
                pred = self.forward(input_tensor)
                predicted_prices = self.inverse_transform(pred)
                logger.info(f"预测未来24小时价格: {predicted_prices.tolist()}")
                return predicted_prices.tolist()
        except Exception as e:
            logger.error(f"价格预测错误: {e}")
            return None

    def train_model(self, _train_data, _train_targets, time_steps=10, epochs=100, batch_size=32, validation_split=0.2):
        """训练价格预测模型"""
        try:
            data_array = np.array(_train_data)
            target_array = np.array(_train_targets)
            self.scaler.fit(data_array[:, 0].reshape(-1, 1))

            data_seq, target_seq = [], []
            for i in range(len(data_array) - time_steps):
                data_seq.append(data_array[i:i + time_steps])
                target_seq.append(target_array[i + time_steps - 1])
            data_seq, target_seq = np.array(data_seq), np.array(target_seq)

            split_idx = int(len(data_seq) * (1 - validation_split))
            train_data = data_seq[:split_idx]
            val_data = data_seq[split_idx:]
            train_targets = target_seq[:split_idx]
            val_targets = target_seq[split_idx:]

            train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
            train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
            val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
            val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)

            train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_targets_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            for epoch in range(epochs):
                self.train()
                total_loss = 0
                for batch_data, batch_targets in train_loader:
                    batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
                    optimizer.zero_grad()
                    output = self.forward(batch_data)
                    loss = criterion(output, batch_targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                self.eval()
                with torch.no_grad():
                    val_output = self.forward(val_data_tensor.to(device))
                    val_loss = criterion(val_output, val_targets_tensor.to(device))

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")

            torch.save(self.state_dict(), self.model_path)
            logger.info("价格预测模型训练完成并保存")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_data, new_targets, time_steps=10, epochs=10):
        try:
            self.load_state_dict(torch.load(self.model_path))
            new_data_array = np.array(new_data)
            new_target_array = np.array(new_targets)
            new_data_seq = []
            new_target_seq = []
            for i in range(len(new_data_array) - time_steps):
                new_data_seq.append(new_data_array[i:i + time_steps])
                new_target_seq.append(new_target_array[i + time_steps - 1])

            data_tensor = torch.tensor(np.array(new_data_seq), dtype=torch.float32)
            target_tensor = torch.tensor(np.array(new_target_seq), dtype=torch.float32)

            optimizer = optim.Adam(self.parameters(), lr=0.0001)
            criterion = nn.MSELoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)

            self.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self.forward(data_tensor.to(device))
                loss = criterion(output, target_tensor.to(device))
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    logger.info(f"Update Epoch {epoch}, Loss: {loss.item():.4f}")

            torch.save(self.state_dict(), self.model_path)
            logger.info("模型在线更新完成")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")


if __name__ == "__main__":
    np.random.seed(42)
    prices = np.random.normal(150, 10, 1000)
    data = [[p, 0.5, 0.2, 1000, 200] for p in prices]
    targets = [prices[i:i + 24] for i in range(len(prices) - 24)]

    model = PricePredictor()
    model.train_model(data, targets, time_steps=10, epochs=50)

    sample_data = data[-10:]
    predicted = model.predict(sample_data)
    print(f"预测未来24小时价格: {predicted}")