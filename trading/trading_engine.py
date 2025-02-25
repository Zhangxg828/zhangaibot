from solana.rpc.api import Client
from .exchange_api import GateIOAPI
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
        self.gateio = GateIOAPI(self.config["gateio_api"]["api_key"], self.config["gateio_api"]["secret"])
        self.nlp = NLPInteraction()
        self.total_funds = 1000
        self.daily_trades = 0

    def execute_trade(self, token, score, amount, price, sentiment=0.5, holder_growth=0.2):
        params = self.config["trading_params"]
        decision = {"action": "hold", "amount": 0, "reason": "默认持有"}
        trading_data = {"score": score, "price": price, "sentiment": sentiment, "holder_growth": holder_growth,
                        "decision": decision}

        if self.daily_trades >= params["daily_trade_limit"]:
            decision["reason"] = "达到每日交易次数限制"
            logger.info(f"交易提示: {token} - {decision['reason']}")
            self.log_decision(token, trading_data)
            return decision

        if score > 0.8:
            current_price = self.gateio.get_price(f"{token}/USDT")
            take_profit = current_price * (1 + params["take_profit_percentage"] / 100)
            stop_loss = current_price * (1 - params["stop_loss_percentage"] / 100)
            if amount <= self.total_funds * params["max_position"] / 100:
                self.gateio.futures_trade(
                    f"{token}/USDT:USDT", "buy", amount,
                    leverage=params["leverage"], take_profit=take_profit, stop_loss=stop_loss
                )
                decision.update({"action": "buy", "amount": amount, "reason": "高评分代币"})
                logger.info(f"交易提示: {token} - 买入 {amount} 单位，评分 {score:.2f}")
                self.daily_trades += 1
            else:
                decision["reason"] = "超过最大持仓限制"
                logger.info(f"交易提示: {token} - {decision['reason']}")
        else:
            decision["reason"] = "评分不足0.8"
            logger.info(f"交易提示: {token} - {decision['reason']}")

        self.log_decision(token, trading_data)

        # 记录交易到数据库
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "token": token,
            "amount": amount,
            "price": price,
            "action": decision["action"],
            "reason": decision["reason"]
        }
        db = Database(self.config["database"])
        db.insert_trade_record(trade_record)
        db.close()

        return decision

    def log_decision(self, token, trading_data):
        logger.debug(f"代币: {token}, 评分: {trading_data['score']:.2f}, 决定: {trading_data['decision']}")

    def interact(self, query, trading_data):
        context = self.nlp.generate_context(trading_data)
        response = self.nlp.process_query(query, context)
        logger.info(f"用户查询: {query}, 回答: {response}")
        print(f"AI回答: {response}")
        return response