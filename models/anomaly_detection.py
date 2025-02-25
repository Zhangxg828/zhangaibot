import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import joblib
import os
import yaml
from utils.db import Database
from utils.logger import setup_logger

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
    # ... (preprocess_data, detect, train_model, update_model)