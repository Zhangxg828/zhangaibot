import asyncio
import websockets
import json
from kafka import KafkaProducer
from utils.logger import setup_logger
import yaml

logger = setup_logger("binance_consumer")


class BinanceConsumer:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        """初始化Binance WebSocket消费者"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,
            max_block_ms=5000
        )
        self.enabled = self.config.get("binance_api", {}).get("enabled", False)
        self.symbols = ["BTCUSDT", "ETHUSDT"]  # 默认监控BTC和ETH，可配置扩展
        logger.info("Binance WebSocket消费者初始化完成")

    async def fetch_realtime_data(self):
        """通过WebSocket获取K线、价格和深度数据"""
        if not self.enabled:
            logger.info("Binance WebSocket未启用，跳过")
            return

        streams = [f"{symbol.lower()}@kline_1m" for symbol in self.symbols] + \
                  [f"{symbol.lower()}@ticker" for symbol in self.symbols] + \
                  [f"{symbol.lower()}@depth20" for symbol in self.symbols]
        stream_url = f"{self.ws_url}/{'/'.join(streams)}"

        async with websockets.connect(stream_url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    await self.process_message(data)
                except Exception as e:
                    logger.error(f"Binance WebSocket错误: {e}")
                    await asyncio.sleep(5)  # 重连等待

    async def process_message(self, data):
        """处理Binance WebSocket数据并推送至Kafka"""
        try:
            if "k" in data:  # K线数据
                kline = data["k"]
                processed = {
                    "symbol": kline["s"],
                    "type": "kline",
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                    "timestamp": kline["t"] / 1000
                }
            elif "b" in data and "a" in data:  # 深度数据
                processed = {
                    "symbol": data["s"],
                    "type": "depth",
                    "bids": [[float(b[0]), float(b[1])] for b in data["b"][:5]],  # 前5个买单
                    "asks": [[float(a[0]), float(a[1])] for a in data["a"][:5]],  # 前5个卖单
                    "timestamp": data["E"] / 1000
                }
            elif "c" in data:  # 最新价格
                processed = {
                    "symbol": data["s"],
                    "type": "price",
                    "price": float(data["c"]),
                    "timestamp": data["E"] / 1000
                }
            else:
                return

            self.producer.send("binance_stream", value=processed)
            logger.debug(f"推送Binance数据至Kafka: {processed['symbol']} - {processed['type']}")
        except Exception as e:
            logger.error(f"处理消息错误: {e}")

    def run(self):
        """运行Binance WebSocket消费者"""
        if self.enabled:
            asyncio.run(self.fetch_realtime_data())
        else:
            logger.info("Binance WebSocket已禁用")


if __name__ == "__main__":
    consumer = BinanceConsumer()
    consumer.run()