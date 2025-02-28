import requests
import json
from kafka import KafkaProducer
import time
from utils.logger import setup_logger
import yaml

logger = setup_logger("glassnode_consumer")


class GlassnodeConsumer:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.api_url = "https://api.glassnode.com/v1/metrics"
        self.api_key = self.config["glassnode_api"]["api_key"]
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3
        )
        self.enabled = self.config["glassnode_api"]["enabled"]
        self.assets = ["BTC", "ETH"]  # 默认监控BTC和ETH
        logger.info("Glassnode消费者初始化完成")

    def fetch_onchain_data(self, asset, metric):
        """获取Glassnode链上数据"""
        if not self.enabled:
            logger.info("Glassnode消费者未启用，跳过")
            return None

        try:
            url = f"{self.api_url}/{metric}"
            params = {
                "api_key": self.api_key,
                "a": asset,
                "f": "JSON",
                "i": "24h"  # 按天获取
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data[-1]  # 获取最新数据点
        except Exception as e:
            logger.error(f"获取Glassnode数据失败: {e}")
            return None

    def run(self):
        """运行Glassnode消费者"""
        while True:
            if self.enabled:
                for asset in self.assets:
                    try:
                        # 示例指标：交易活跃度、持仓变化
                        tx_count = self.fetch_onchain_data(asset, "transactions/count")
                        balance_change = self.fetch_onchain_data(asset, "addresses/balance_change")
                        if tx_count and balance_change:
                            data = {
                                "asset": asset,
                                "tx_count": tx_count["v"],
                                "balance_change": balance_change["v"],
                                "timestamp": tx_count["t"]
                            }
                            self.producer.send("glassnode_stream", value=data)
                            logger.info(f"推送Glassnode数据: {asset}")
                    except Exception as e:
                        logger.error(f"处理 {asset} 数据错误: {e}")
            else:
                logger.info("Glassnode消费者已禁用")
            time.sleep(3600)  # 每小时更新一次


if __name__ == "__main__":
    consumer = GlassnodeConsumer()
    consumer.run()