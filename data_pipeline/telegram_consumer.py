import asyncio
from telethon import TelegramClient
from kafka import KafkaProducer
from utils.logger import setup_logger
import yaml
import json

logger = setup_logger("telegram_consumer")


class TelegramConsumer:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        """初始化Telegram社群数据消费者"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 初始化Telegram客户端，使用telethon连接Telegram API
        self.client = TelegramClient(
            "telegram_session",
            self.config["telegram_api"]["api_id"],
            self.config["telegram_api"]["api_hash"]
        )

        # Kafka生产者
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,
            max_block_ms=5000
        )

        self.enabled = self.config["telegram_api"]["enabled"]
        self.channels = self.config["telegram_api"]["channels"]
        logger.info("Telegram消费者初始化完成")

    async def fetch_messages(self):
        """从Telegram频道或群组获取社群消息"""
        if not self.enabled:
            logger.info("Telegram消费者未启用，跳过运行")
            return

        try:
            async with self.client:
                for channel in self.channels:
                    try:
                        async for message in self.client.iter_messages(channel, limit=100):
                            if message.text:  # 仅处理包含文本的消息
                                data = {
                                    "channel": str(channel),
                                    "text": message.text,
                                    "timestamp": message.date.timestamp()
                                }
                                self.producer.send("telegram_stream", value=data)
                                logger.debug(f"推送Telegram消息: {channel} - {message.text[:50]}")
                    except Exception as e:
                        logger.error(f"从 {channel} 获取消息失败: {e}")
        except Exception as e:
            logger.error(f"Telegram客户端运行错误: {e}")

        self.producer.flush()  # 确保所有消息推送完成

    def run(self):
        """运行Telegram消费者"""
        if self.enabled:
            try:
                asyncio.run(self.fetch_messages())
            except Exception as e:
                logger.error(f"Telegram消费者启动失败: {e}")
        else:
            logger.info("Telegram消费者已禁用")


if __name__ == "__main__":
    consumer = TelegramConsumer()
    consumer.run()