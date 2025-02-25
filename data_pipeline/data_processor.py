from utils.db import Database
import yaml

class DataProcessor:
    def __init__(self):
        with open("data_pipeline/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.db = Database(config["database"])

    def process_chain_data(self, data):
        features = {
            "timestamp": data["timestamp"],
            "token": data.get("token", "unknown"),
            "liquidity": data.get("liquidity", 0),
            "holder_count": data.get("holders", 0),
            "tx_volume": data.get("volume", 0),
            "creator_address": data.get("creator_address", "")
        }
        self.db.insert_chain_data(features)
        return {k: v for k, v in features.items() if k != "timestamp"}

    def process_social_data(self, data):
        features = {
            "timestamp": data["timestamp"],
            "source": data.get("source", "unknown"),
            "text": data["text"],
            "sentiment": data.get("sentiment", 0.0)
        }
        self.db.insert_social_data(features)
        return features