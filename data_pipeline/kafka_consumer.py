from kafka import KafkaConsumer
import json
import time
from utils.logger import setup_logger
import yaml
from solana.rpc.websocket_api import connect
from solana.publickey import Pubkey  # 正确导入Pubkey，solana-py 0.36.6支持

logger = setup_logger("kafka_consumer")


class KafkaDataConsumer:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        """初始化Kafka消费者，支持Solana、Pump链和Telegram实时数据"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Kafka消费者配置
        self.consumer = KafkaConsumer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="crypto_trading_group",
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            max_poll_records=100
        )

        # 订阅主题
        self.topics = ["twitter_stream", "solana_stream", "pump_stream", "binance_stream",
                       "discord_stream", "telegram_stream", "glassnode_stream", "lunarcrush_stream"]
        self.consumer.subscribe(self.topics)

        # Solana WebSocket URL
        self.solana_ws_url = self.config["solana_rpc"]

        logger.info("Kafka消费者初始化完成")

    async def connect_solana_websocket(self):
        """连接Solana WebSocket，订阅交易数据"""
        try:
            async with connect(self.solana_ws_url) as ws:
                # 订阅账户变化，使用Pubkey类型
                token_program_id = Pubkey.from_string(
                    "TokenkegQfeZyiNwAJbNbGKpfDzkL4Y39cXdzG2Yk")  # Solana Token Program ID
                await ws.account_subscribe(token_program_id)
                logger.info("Solana WebSocket连接成功")
                # 处理订阅消息
                async for msg in ws:
                    logger.debug(f"Solana WebSocket消息: {msg}")
        except Exception as e:
            logger.error(f"Solana WebSocket连接失败: {e}")

    @staticmethod
    def process_message(msg):
        """处理Kafka消息（静态方法）"""
        try:
            topic = msg.topic
            data = msg.value
            if topic in ["twitter_stream", "discord_stream", "telegram_stream"]:
                return {"type": "social", "text": data["text"], "timestamp": data["timestamp"]}
            elif topic in ["solana_stream", "pump_stream"]:
                return {"type": "chain", "token": data["token"], "liquidity": data["liquidity"],
                        "volume": data["volume"]}
            elif topic == "binance_stream":
                if data["type"] in ["kline", "price"]:
                    return {"type": data["type"], "price": data.get("close", data.get("price", 0)),
                            "timestamp": data["timestamp"]}
            elif topic == "glassnode_stream":
                return data
            elif topic == "lunarcrush_stream":
                return {"fear_index": data["fear_index"], "timestamp": data["timestamp"]}
            logger.debug(f"处理消息: {topic} - {data}")
        except Exception as e:
            logger.error(f"消息处理错误: {e}")
            return None

    def run(self):
        """运行Kafka消费者，处理实时数据流"""
        import asyncio
        asyncio.run(self.connect_solana_websocket())  # 启动Solana WebSocket

        while True:
            try:
                # 轮询Kafka消息
                messages = self.consumer.poll(timeout_ms=1000)
                for topic_partition, partition_msgs in messages.items():
                    for msg in partition_msgs:
                        processed_data = self.process_message(msg)
                        if processed_data:
                            logger.info(f"消费数据: {processed_data}")

                time.sleep(1)  # 控制轮询频率
            except KeyboardInterrupt:
                logger.info("Kafka消费者停止")
                break
            except Exception as e:
                logger.error(f"运行错误: {e}")
                time.sleep(10)  # 出错后等待10秒重试


if __name__ == "__main__":
    consumer = KafkaDataConsumer()
    consumer.run()