import xgboost as xgb
import pandas as pd
import numpy as np
import os
from utils.logger import setup_logger
from utils.data_utils import normalize_data

logger = setup_logger("token_scoring")


class TokenScoring:
    def __init__(self, model_path="models/token_scoring_model.json", blacklist_path="models/blacklist.txt"):
        self.model_path = model_path
        self.blacklist_path = blacklist_path
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        self.weights = {
            "liquidity": 0.30,  # 流动性占比30%
            "social_sentiment": 0.25,  # 社交情绪占比25%
            "holder_growth": 0.20,  # 持有人增长占比20%
            "creator_reputation": 0.15,  # 发起人信誉占比15%
            "other": 0.10  # 其他因素占比10%
        }
        self.blacklist = self.load_blacklist()

        if os.path.exists(model_path):
            try:
                self.model.load_model(model_path)
                logger.info("成功加载预训练代币评分模型")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
        else:
            logger.warning("未找到预训练模型，需要训练")

    def load_blacklist(self):
        """加载黑名单地址"""
        blacklist = set()
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, "r") as f:
                blacklist = set(line.strip() for line in f if line.strip())
            logger.info(f"加载黑名单，包含 {len(blacklist)} 个地址")
        return blacklist

    def update_blacklist(self, address):
        """更新黑名单"""
        self.blacklist.add(address)
        with open(self.blacklist_path, "a") as f:
            f.write(f"{address}\n")
        logger.info(f"添加地址 {address} 到黑名单")

    def preprocess_features(self, features):
        """预处理特征数据"""
        try:
            # 确保所有必要特征存在，缺失值填0
            required_features = ["liquidity", "social_sentiment", "holder_growth", "creator_reputation", "tx_volume"]
            processed = {key: features.get(key, 0.0) for key in required_features}

            # 归一化特征
            df = pd.DataFrame([processed])
            normalized_df = normalize_data(df)

            # 检查黑名单
            creator_address = features.get("creator_address", "")
            if creator_address in self.blacklist:
                logger.warning(f"地址 {creator_address} 在黑名单中，评分强制为0")
                return None, 0.0
