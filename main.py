import numpy as np
from trading.trading_engine import TradingEngine
from trading.risk_manager import RiskManager
from models.token_scoring import TokenScoring
from models.price_prediction import PricePredictor
from models.sentiment_analysis import SentimentAnalyzer
from models.anomaly_detection import AnomalyDetector
from models.rl_trading import RLTrader
from models.nlp_interaction import NLPInteraction
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

    # 生成加密密钥
    encryption_key = generate_key()

    # 模拟实时数据（实际应从data_processor.py获取）
    trading_data = {
        "score": 0.0,
        "price": 150.0,
        "sentiment": 0.5,
        "holder_growth": 0.2,
        "decision": {"action": "hold", "reason": "初始化"},
        "features": {"liquidity": 800, "social_sentiment": 0.6, "holder_growth": 0.25, "creator_reputation": 0.7,
                     "tx_volume": 150},
        "price_history": [[150 + i, 0.5, 0.2, 1000, 200] for i in range(10)],
        "texts": ["This coin is awesome!", "Total scam"],
        "anomaly_data": [
            {"volume": 6000, "price_change": 0.6, "liquidity_change": -3000, "tx_count": 5, "address": "addr3"},
            {"volume": 800, "price_change": 0.02, "liquidity_change": 50, "tx_count": 40, "address": "addr1"}
        ]
    }

    # 加密敏感数据（示例）
    encrypted_price = encrypt_key(str(trading_data["price"]).encode(), encryption_key)
    decrypted_price = float(decrypt_key(encrypted_price, encryption_key).decode())
    logger.info(f"加密价格: {encrypted_price}, 解密价格: {decrypted_price}")

    # 1. 代币评分
    trading_data["score"] = scorer.score_token(trading_data["features"])
    logger.info(f"代币评分: {trading_data['score']:.3f}")

    # 2. 情绪分析
    sentiment_scores = sentiment_analyzer.analyze(trading_data["texts"])
    trading_data["sentiment"] = float(np.mean(sentiment_scores))
    logger.info(f"平均情绪得分: {trading_data['sentiment']:.3f}")

    # 3. 价格预测
    predicted_prices = predictor.predict(trading_data["price_history"])
    if predicted_prices:
        logger.info(f"预测未来24小时价格: {predicted_prices[:5]}...")

    # 4. 异常检测
    anomalies = anomaly_detector.detect(trading_data["anomaly_data"])
    if any(anomalies):
        logger.warning("检测到异常交易，跳过执行")
        trading_data["decision"] = {"action": "skip", "reason": "anomaly detected"}
    else:
        # 5. 强化学习决策
        rl_state = [
            trading_data["score"],
            trading_data["price"],
            trading_data["sentiment"],
            trading_data["holder_growth"],
            trading_data["features"]["tx_volume"]
        ]
        rl_action = rl_trader.decide(rl_state)
        logger.info(f"RL决策: {rl_action}")

        # 6. 检查风险并执行交易
        if rl_action in ["buy", "sell"]:
            params = {"max_position": 20, "circuit_breaker_loss": 10}
            if risk_manager.check_risk(trade_amount=10, current_loss=0, risk_params=params):  # 修复参数名
                decision = engine.execute_trade(
                    "SOL", trading_data["score"], 10, trading_data["price"],
                    sentiment=trading_data["sentiment"], holder_growth=trading_data["holder_growth"]
                )
                trading_data["decision"] = decision
                risk_manager.update_loss(0)  # 示例，未计算实际亏损

    # 7. 自然语言交互
    while True:
        query = input("请输入问题（输入'exit'退出）: ")
        if query.lower() == "exit":
            break
        context = nlp.generate_context(trading_data)
        response = nlp.process_query(query, trading_data, context=context)
        print(f"回答: {response}")


if __name__ == "__main__":
    main()