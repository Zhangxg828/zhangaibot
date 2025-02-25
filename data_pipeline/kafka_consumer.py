from kafka import KafkaConsumer
import json
import yaml
from utils.logger import setup_logger

logger = setup_logger("kafka_consumer")

class KafkaDataConsumer:
    def __init__(self):
        with open("data_pipeline/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.consumer = KafkaConsumer(
            "solana_transactions", "twitter_stream",
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            auto_offset_reset="latest",
            value_deserializer=lambda x: json.loads(x.decode("utf-8"))
        )

    def consume_data(self):
        """从Kafka消费数据并处理"""
        for message in self.consumer:
            topic = message.topic
            data = message.value
            if topic == "solana_transactions":
                logger.info(f"链上数据: {data}")
            elif topic == "twitter_stream":
                logger.info(f"社交数据: {data}")
            # 此处可添加数据处理逻辑，传递给data_processor.py

if __name__ == "__main__":
    consumer = KafkaDataConsumer()
    consumer.consume_data()