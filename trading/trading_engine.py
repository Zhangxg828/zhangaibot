from solana.rpc.api import Client
from trading.exchange_api import GateIOAPI
import yaml
from datetime import datetime
from models.nlp_interaction import NLPInteraction
from utils.db import Database
from utils.logger import setup_logger

logger = setup_logger("trading_engine")

class TradingEngine:
    def __init__(self):
        with open("data_pipeline/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.solana_client = Client(self.config["solana_rpc"])
        self.gateio = GateIOAPI(self.config["gateio_api"]["api_key"], self.config["gateio_api"]["secret"])  # gateio是正确命名
        self.nlp = NLPInteraction()
        self.total_funds = 1000
        self.daily_trades = 0

    def execute_trade(self, token, score, amount, price, sentiment=0.5, holder_growth=0.2):
        try:
            params = self.config["trading_params"]
            trade_decision = {"action": "hold", "amount": 0, "reason": "默认持有"}  # 重命名避免shadowing
            trading_data = {"score": score, "price": price, "sentiment": sentiment, "holder_growth": holder_growth, "decision": trade_decision}

            if self.daily_trades >= params["daily_trade_limit"]:
                trade_decision["reason"] = "达到每日交易次数限制"
                logger.info(f"交易提示: {token} - {trade_decision['reason']}")
                self.log_decision(token, trading_data)
                return trade_decision

            if score > 0.8:
                current_price = self.gateio.get_price(f"{token}/USDT")
                take_profit = current_price * (1 + params["take_profit_percentage"] / 100)
                stop_loss = current_price * (1 - params["stop_loss_percentage"] / 100)
                if amount <= self.total_funds * params["max_position"] / 100:
                    self.gateio.futures_trade(
                        f"{token}/USDT:USDT", "buy", amount,
                        leverage=params["leverage"], take_profit=take_profit, stop_loss=stop_loss
                    )
                    trade_decision.update({"action": "buy", "amount": amount, "reason": "高评分代币"})
                    logger.info(f"交易提示: {token} - 买入 {amount} 单位，评分 {score:.2f}")
                    self.daily_trades += 1
                else:
                    trade_decision["reason"] = "超过最大持仓限制"
                    logger.info(f"交易提示: {token} - {trade_decision['reason']}")
            else:
                trade_decision["reason"] = "评分不足0.8"
                logger.info(f"交易提示: {token} - {trade_decision['reason']}")

            self.log_decision(token, trading_data)
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "token": token,
                "amount": amount,
                "price": price,
                "action": trade_decision["action"],
                "reason": trade_decision["reason"]
            }
            db = Database(self.config["database"])
            db.insert_trade_record(trade_record)
            db.close()
            return trade_decision
        except Exception as e:
            logger.error(f"交易执行错误: {e}")
            return {"action": "error", "reason": str(e)}

    @staticmethod
    def log_decision(token, trading_data):
        """记录交易决策日志"""
        logger.debug(f"代币: {token}, 评分: {trading_data['score']:.2f}, 决定: {trading_data['decision']}")

    def interact(self, query, trading_data):
        context = self.nlp.generate_context(trading_data)  # 计算context
        response = self.nlp.process_query(query, trading_data, context=context)  # 使用context
        logger.info(f"用户查询: {query}, 回答: {response}")
        print(f"AI回答: {response}")
        return response

if __name__ == "__main__":
    engine = TradingEngine()
    decision = engine.execute_trade("SOL", 0.85, 10, 150.0)
    engine.interact("为什么买入这个代币？", {"score": 0.85, "price": 150.0, "sentiment": 0.7, "holder_growth": 0.3, "decision": decision})