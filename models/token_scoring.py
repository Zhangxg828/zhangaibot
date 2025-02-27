import xgboost as xgb
import pandas as pd
import numpy as np  # 保留，因np.clip使用
import os
from utils.logger import setup_logger
from utils.data_utils import normalize_data

logger = setup_logger("token_scoring")


class TokenScoring:
    def __init__(self, model_path="models/token_scoring_model.json", blacklist_path="models/blacklist.txt"):
        self.model_path = model_path
        self.blacklist_path = blacklist_path
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",  # noqa: squarederror is correct
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        self.weights = {
            "liquidity": 0.30,
            "social_sentiment": 0.25,
            "holder_growth": 0.20,
            "creator_reputation": 0.15,
            "other": 0.10
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
        blacklist = set()
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, "r") as f:
                blacklist = set(line.strip() for line in f if line.strip())
            logger.info(f"加载黑名单，包含 {len(blacklist)} 个地址")
        return blacklist

    def update_blacklist(self, address):
        self.blacklist.add(address)
        with open(self.blacklist_path, "a") as f:
            f.write(f"{address}\n")
        logger.info(f"添加地址 {address} 到黑名单")

    def preprocess_features(self, features):
        try:
            required_features = ["liquidity", "social_sentiment", "holder_growth", "creator_reputation", "tx_volume"]
            processed = {key: features.get(key, 0.0) for key in required_features}
            df = pd.DataFrame([processed])
            creator_address = features.get("creator_address", "")
            if creator_address in self.blacklist:
                logger.warning(f"地址 {creator_address} 在黑名单中，评分强制为0")
                return None, 0.0
            return normalize_data(df).iloc[0].to_dict(), None  # 优化，去除normalized_df
        except Exception as e:
            logger.error(f"特征预处理错误: {e}")
            return None, 0.5

    def score_token(self, features):
        try:
            processed_features, forced_score = self.preprocess_features(features)
            if processed_features is None:
                return forced_score
            df = pd.DataFrame([processed_features])
            base_score = self.model.predict(df)[0]
            weighted_score = (
                    processed_features["liquidity"] * self.weights["liquidity"] +
                    processed_features["social_sentiment"] * self.weights["social_sentiment"] +
                    processed_features["holder_growth"] * self.weights["holder_growth"] +
                    processed_features["creator_reputation"] * self.weights["creator_reputation"] +
                    processed_features["tx_volume"] * self.weights["other"]
            )
            final_score = np.clip((base_score + weighted_score) / 2, 0, 1)
            logger.info(f"代币评分: {final_score:.3f}, 特征: {processed_features}")
            return final_score
        except Exception as e:
            logger.error(f"评分错误: {e}")
            return 0.5  # 添加except修复语法

    def train_model(self, data, target, validation_split=0.2):
        try:
            x = pd.DataFrame(data)
            y = pd.Series(target)
            split_idx = int(len(x) * (1 - validation_split))
            x_train, x_val = x.iloc[:split_idx], x.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=10, verbose=False)
            self.model.save_model(self.model_path)
            val_score = self.model.score(x_val, y_val)
            logger.info(f"模型训练完成，验证集R²得分: {val_score:.3f}")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_data, new_target):
        try:
            x = pd.DataFrame(new_data)
            y = pd.Series(new_target)
            self.model.fit(x, y, xgb_model=self.model_path)
            self.model.save_model(self.model_path)
            logger.info("模型在线更新完成")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")


if __name__ == "__main__":
    scoring = TokenScoring()
    train_data = [
        {"liquidity": 1000, "social_sentiment": 0.7, "holder_growth": 0.3, "creator_reputation": 0.8, "tx_volume": 200},
        {"liquidity": 500, "social_sentiment": 0.2, "holder_growth": 0.1, "creator_reputation": 0.4, "tx_volume": 100}
    ]
    train_target = [0.9, 0.3]
    scoring.train_model(train_data, train_target)
    sample_features = {
        "liquidity": 800,
        "social_sentiment": 0.6,
        "holder_growth": 0.25,
        "creator_reputation": 0.7,
        "tx_volume": 150,
        "creator_address": "example_address"
    }
    score = scoring.score_token(sample_features)
    print(f"代币评分: {score:.3f}")
    scoring.update_blacklist("example_address")
    score_after_blacklist = scoring.score_token(sample_features)
    print(f"黑名单后评分: {score_after_blacklist:.3f}")