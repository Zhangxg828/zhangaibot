import requests
import json
from kafka import KafkaProducer
import time
from utils.logger import setup_logger
import yaml

logger = setup_logger("lunarcrush_consumer")


class LunarCrushConsumer:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.api_url = "https://api.lunarcrush.com/v2"
        self.api_key = self.config["lunarcrush_api"]["api_key"]
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3
        )
        self.enabled = self.config["lunarcrush_api"]["enabled"]
        logger.info("LunarCrush消费者初始化完成")

    def fetch_fear_index(self):
        """获取LunarCrush市场恐惧指数"""
        if not self.enabled:
            logger.info("LunarCrush消费者未启用，跳过")
            return None

        try:
            url = f"{self.api_url}/market"
            params = {
                "key": self.api_key,
                "type": "fear_greed"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()["data"]
            return data["fear_greed"]
        except Exception as e:
            logger.error(f"获取LunarCrush数据失败: {e}")
            return None

    def run(self):
        """运行LunarCrush消费者"""
        while True:
            if self.enabled:
                try:
                    fear_index = self.fetch_fear_index()
                    if fear_index:
                        data = {
                            "fear_index": fear_index,
                            "timestamp": time.time()
                        }
                        self.producer.send("lunarcrush_stream", value=data)
                        logger.info(f"推送LunarCrush恐惧指数: {fear_index}")
                except Exception as e:
                    logger.error(f"运行错误: {e}")
            else:
                logger.info("LunarCrush消费者已禁用")
            time.sleep(3600)  # 每小时更新


if __name__ == "__main__":
    consumer = LunarCrushConsumer()
    consumer.run()