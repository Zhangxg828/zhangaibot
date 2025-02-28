import numpy as np
import asyncio
from trading.trading_engine import TradingEngine
from trading.risk_manager import RiskManager
from models.token_scoring import TokenScoring
from models.price_prediction import PricePredictor
from models.sentiment_analysis import SentimentAnalyzer
from models.anomaly_detection import AnomalyDetector
from models.rl_trading import RLTrader
from models.nlp_interaction import NLPInteraction
from data_pipeline.kafka_consumer import KafkaDataConsumer
from data_pipeline.binance_consumer import BinanceConsumer
from data_pipeline.discord_consumer import DiscordConsumer  # 更新为discord.py支持
from data_pipeline.telegram_consumer import TelegramConsumer
from data_pipeline.glassnode_consumer import GlassnodeConsumer
from data_pipeline.lunarcrush_consumer import LunarCrushConsumer
from utils.logger import setup_logger
from utils.security import encrypt_key, decrypt_key, generate_key

logger = setup_logger("main")


def main():
    # 初始化所有模块
    engine = TradingEngine()
    risk_manager = RiskManager(total_funds=1000)
    scorer = TokenScoring()
    predictor = PricePredictor()
    sentiment_analyzer = SentimentAnalyzer()
    anomaly_detector = AnomalyDetector()
    rl_trader = RLTrader()
    nlp = NLPInteraction()
    kafka_consumer = KafkaDataConsumer()
    binance_consumer = BinanceConsumer()
    discord_consumer = DiscordConsumer()  # 使用discord.py实现
    telegram_consumer = TelegramConsumer()
    glassnode_consumer = GlassnodeConsumer()
    lunarcrush_consumer = LunarCrushConsumer()

    # 生成加密密钥
    encryption_key = generate_key()

    # 启动异步消费者
    asyncio.ensure_future(binance_consumer.fetch_realtime_data())
    asyncio.ensure_future(telegram_consumer.fetch_messages())
    asyncio.ensure_future(discord_consumer.run(discord_consumer.bot_token))  # 使用discord.py的start方法
    glassnode_consumer.run()
    lunarcrush_consumer.run()

    # 初始交易数据
    trading_data = {
        "score": 0.0,
        "price": 0.0,
        "sentiment": 0.5,
        "holder_growth": 0.0,
        "decision": {"action": "hold", "reason": "初始化"},
        "features": {"liquidity": 0, "social_sentiment": 0.0, "holder_growth": 0.0, "creator_reputation": 0.0,
                     "tx_volume": 0},
        "price_history": [],
        "texts": [],
        "anomaly_data": [],
        "fear_index": 50
    }

    logger.info("开始实时数据处理")
    while True:
        try:
            messages = kafka_consumer.consumer.poll(timeout_ms=1000)
            for topic_partition, partition_msgs in messages.items():
                for msg in partition_msgs:
                    processed_data = kafka_consumer.process_message(msg)
                    if processed_data:
                        if processed_data["type"] == "social":
                            trading_data["texts"].append(processed_data["text"])
                        elif processed_data["type"] == "chain":
                            trading_data["features"].update({
                                "liquidity": processed_data["liquidity"],
                                "tx_volume": processed_data["volume"],
                                "token": processed_data.get("token", "UNKNOWN")
                            })
                            trading_data["anomaly_data"].append(processed_data)
                        elif processed_data["type"] in ["kline", "price"]:
                            trading_data["price"] = processed_data.get("close", processed_data.get("price",
                                                                                                   trading_data[
                                                                                                       "price"]))
                            trading_data["price_history"].append([trading_data["price"], 0.5, 0.2, 1000, 200])
                        elif "fear_index" in processed_data:
                            trading_data["fear_index"] = processed_data["fear_index"]

            if trading_data["features"]["token"]:
                encrypted_price = encrypt_key(str(trading_data["price"]).encode(), encryption_key)
                decrypted_price = float(decrypt_key(encrypted_price, encryption_key).decode())
                logger.info(f"加密价格: {encrypted_price}, 解密价格: {decrypted_price}")

                trading_data["score"] = scorer.score_token(trading_data["features"])
                logger.info(f"代币评分: {trading_data['score']:.3f}")

                if trading_data["texts"]:
                    sentiment_scores = sentiment_analyzer.analyze(trading_data["texts"],
                                                                  source="discord" if "discord" in
                                                                                      trading_data["texts"][
                                                                                          0] else "telegram")
                    trading_data["sentiment"] = float(np.mean(sentiment_scores))
                    logger.info(f"平均情绪得分 (Discord/Telegram): {trading_data['sentiment']:.3f}")

                if len(trading_data["price_history"]) >= 10:
                    predicted_prices = predictor.predict(trading_data["price_history"][-10:])
                    if predicted_prices:
                        logger.info(f"预测未来24小时价格: {predicted_prices[:5]}...")

                anomalies = anomaly_detector.detect(trading_data["anomaly_data"])
                if any(anomalies):
                    logger.warning("检测到异常交易，跳过执行")
                    trading_data["decision"] = {"action": "skip", "reason": "anomaly detected"}
                else:
                    rl_state = [
                        trading_data["score"],
                        trading_data["price"],
                        trading_data["sentiment"],
                        trading_data["holder_growth"],
                        trading_data["features"]["tx_volume"],
                        trading_data["fear_index"] / 100
                    ]
                    rl_action = rl_trader.decide(rl_state)
                    logger.info(f"RL决策: {rl_action}")

                    if rl_action in ["buy", "sell"]:
                        params = {"max_position": 20, "circuit_breaker_loss": 10}
                        if risk_manager.check_risk(trade_amount=10, current_loss=0, risk_params=params):
                            decision = engine.execute_trade(
                                trading_data["features"]["token"], trading_data["score"], 10, trading_data["price"],
                                sentiment=trading_data["sentiment"], holder_growth=trading_data["holder_growth"]
                            )
                            trading_data["decision"] = decision
                            risk_manager.update_loss(0)

                trading_data["texts"] = []
                trading_data["anomaly_data"] = []

            query = input("请输入问题（输入'exit'退出）: ")
            if query.lower() == "exit":
                break
            context = nlp.generate_context(trading_data)
            response = nlp.process_query(query, trading_data, context=context)
            print(f"回答: {response}")

        except KeyboardInterrupt:
            logger.info("程序停止")
            break
        except Exception as e:
            logger.error(f"运行错误: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()