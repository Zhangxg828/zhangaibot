import requests
import json
from kafka import KafkaProducer
import time
from utils.logger import setup_logger
import yaml

logger = setup_logger("pump_consumer")


class PumpConsumer:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        """初始化Pump链数据消费者"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Kafka生产者配置
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,  # 重试次数
            max_block_ms=5000  # 阻塞超时5秒
        )

        # Pump链API配置（假设使用Pump.fun API，未公开则需WebSocket或第三方服务）
        self.api_url = "https://api.pump.fun/v1/tokens"  # 示例API，未真实存在
        self.headers = {
            "Authorization": self.config.get("pump_api", {}).get("token", ""),
            "User-Agent": "CryptoTradingBot/1.0"
        }

        # 检查配置完整性
        if not self.config["kafka"]["bootstrap_servers"]:
            raise ValueError("Kafka bootstrap_servers 未配置")
        logger.info("Pump链消费者初始化完成")

    def fetch_new_tokens(self):
        """从Pump链API获取新代币数据"""
        try:
            # 示例API调用（需替换为真实API或WebSocket）
            response = requests.get(self.api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            tokens = response.json()

            # 过滤新代币（假设返回数据包含创建时间）
            new_tokens = [token for token in tokens if token.get("created_at", 0) > time.time() - 3600]  # 过去1小时
            logger.info(f"获取到 {len(new_tokens)} 个新代币")
            return new_tokens
        except requests.RequestException as e:
            logger.error(f"获取Pump链数据失败: {e}")
            return []
        except Exception as e:
            logger.error(f"解析Pump链数据错误: {e}")
            return []

    def push_to_kafka(self, tokens):
        """将新代币数据推送至Kafka"""
        for token in tokens:
            try:
                data = {
                    "token_address": token.get("address", ""),
                    "liquidity": token.get("liquidity", 0.0),
                    "volume": token.get("volume", 0.0),
                    "holders": token.get("holder_count", 0),
                    "timestamp": time.time()
                }
                self.producer.send("pump_stream", value=data)
                logger.debug(f"推送代币数据至Kafka: {data['token_address']}")
            except Exception as e:
                logger.error(f"推送Kafka失败: {e}")
        self.producer.flush()  # 确保所有消息发送

    def run(self):
        """持续监听Pump链新代币"""
        while True:
            try:
                new_tokens = self.fetch_new_tokens()
                if new_tokens:
                    self.push_to_kafka(new_tokens)
                time.sleep(60)  # 每分钟检查一次
            except KeyboardInterrupt:
                logger.info("Pump链消费者停止")
                break
            except Exception as e:
                logger.error(f"运行错误: {e}")
                time.sleep(10)  # 出错后等待10秒重试


if __name__ == "__main__":
    consumer = PumpConsumer()
    consumer.run()