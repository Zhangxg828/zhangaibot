from utils.database import Database
from utils.logger import setup_logger
from utils.security import decrypt_key, generate_key
from trading.exchange_api import GateIOAPI
from trading.risk_manager import RiskManager
from solana.rpc.api import Client as SolanaClient
from solana.publickey import Pubkey  # noqa: IDE未解析，solana-py正确支持
from solana.transaction import Transaction  # noqa: IDE未解析
from solana.system_program import TransferParams, transfer  # noqa: IDE未解析
from solana.keypair import Keypair  # noqa: IDE未解析
import yaml

logger = setup_logger("trading_engine")


class TradingEngine:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        """初始化交易引擎，支持多渠道交易"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 初始化数据库连接
        self.db = Database(config_path)

        # 初始化Gate.io交易所
        self.gateio = GateIOAPI(
            api_key=self.config["gateio_api"]["api_key"],
            secret=self.config["gateio_api"]["secret"]
        )

        # 初始化Solana客户端和钱包
        self.solana_client = SolanaClient(self.config["solana_rpc"])
        encryption_key = generate_key()
        encrypted_private_key = self.config["solana_wallet"]["private_key"]
        private_key = decrypt_key(encrypted_private_key.encode(), encryption_key).decode()
        self.wallet_keypair = Keypair.from_secret_key(bytes.fromhex(private_key))
        logger.info(f"Solana钱包初始化完成: {self.wallet_keypair.public_key}")

        # 初始化风险管理器
        self.risk_manager = RiskManager(total_funds=1000)

        # 交易参数
        self.trading_params = self.config.get("trading_params", {})
        self.leverage = self.trading_params.get("leverage", 5)
        self.take_profit_pct = self.trading_params.get("take_profit_percentage", 5.0)
        self.stop_loss_pct = self.trading_params.get("stop_loss_percentage", 2.0)

        logger.info("交易引擎初始化完成，支持Gate.io、Solana链上和Pump.fun交易")

    def calculate_take_profit_stop_loss(self, price, action):
        """计算止盈止损价格"""
        take_profit = price * (1 + self.take_profit_pct / 100) if action == "buy" else price * (
                    1 - self.take_profit_pct / 100)
        stop_loss = price * (1 - self.stop_loss_pct / 100) if action == "buy" else price * (
                    1 + self.stop_loss_pct / 100)
        return take_profit, stop_loss

    def select_channel(self, token, token_address=None):
        """根据代币信息选择交易渠道"""
        try:
            if token_address and token_address.endswith("pump"):
                return "pumpfun"
            if token_address:
                markets = self.gateio.client.fetch_markets()
                if any(m["symbol"] == f"{token}/USDT" for m in markets):
                    return "gateio"
                return "solana"
            return "gateio"
        except Exception as e:
            logger.error(f"选择交易渠道失败: {e}")
            return "gateio"

    def execute_solana_trade(self, token_address, size, price, action):
        """执行Solana链上交易（通过DEX）"""
        try:
            tx = Transaction()
            tx.add(transfer(TransferParams(
                from_pubkey=self.wallet_keypair.public_key,
                to_pubkey=Pubkey.from_string(token_address),
                lamports=int(size * 10 ** 9)
            )))
            tx_id = self.solana_client.send_transaction(tx, self.wallet_keypair).value

            take_profit, stop_loss = self.calculate_take_profit_stop_loss(price, action)
            logger.info(f"Solana链上交易执行: {token_address} - {action} {size} @ {price}, TxID: {tx_id}")
            return tx_id, take_profit, stop_loss
        except Exception as e:
            logger.error(f"Solana链上交易失败: {e}")
            return None, None, None

    def execute_gateio_trade(self, token, size, price, action):
        """执行Gate.io交易（现货或期货）"""
        try:
            contract = f"{token}_USDT"  # Gate.io期货合约格式
            take_profit, stop_loss = self.calculate_take_profit_stop_loss(price, action)

            trade_id = self.gateio.futures_trade(
                contract=contract, side=action, size=size, price=price,
                leverage=self.leverage, take_profit=take_profit, stop_loss=stop_loss
            )
            logger.info(f"Gate.io交易执行: {token} - {action} {size} @ {price}, Trade ID: {trade_id}")
            return trade_id, take_profit, stop_loss
        except Exception as e:
            logger.error(f"Gate.io交易失败: {e}")
            return None, None, None

    def execute_trade(self, token, score, amount, price, sentiment, holder_growth, token_address=None):
        """执行交易，支持多渠道并设置止盈止损"""
        try:
            # 风险检查
            risk_params = {
                "max_position": self.trading_params.get("max_position", 20.0),
                "circuit_breaker_loss": self.trading_params.get("circuit_breaker_loss", 10.0)
            }
            current_loss = self.risk_manager.get_current_loss()
            if not self.risk_manager.check_risk(trade_amount=amount, current_loss=current_loss,
                                                risk_params=risk_params):
                logger.warning(f"交易取消: {token} - 超出风险限制")
                return None

            # 交易决策逻辑
            if score > 0.8 and sentiment > 0.5:
                action = "buy"
                reason = "高评分和高社区热度"
            elif score < 0.2 or sentiment < -0.5:
                action = "sell"
                reason = "低评分或低社区热度"
            else:
                action = "hold"
                reason = "评分和社区热度中等"
                logger.info(f"交易决策: {token} - {action}，原因: {reason}")
                return {"action": action, "reason": reason, "trade_id": None}

            # 选择交易渠道
            channel = self.select_channel(token, token_address)
            trade_id = None
            take_profit = None
            stop_loss = None

            # 执行交易
            if action in ["buy", "sell"]:
                if channel == "gateio":
                    trade_id, take_profit, stop_loss = self.execute_gateio_trade(token, amount, price, action)
                elif channel in ["solana", "pumpfun"]:
                    trade_id, take_profit, stop_loss = self.execute_solana_trade(token_address or token, amount, price,
                                                                                 action)

                if trade_id:
                    self.risk_manager.update_position(token=token, action=action, amount=amount, price=price)
                    loss = 0 if action == "buy" else (price - price) * amount
                    self.risk_manager.update_loss(loss)

            # 记录交易到数据库
            # noinspection PyTypeChecker
            query = """
            INSERT INTO trades (token, action, amount, price, sentiment, score, holder_growth, trade_id, channel, take_profit_price, stop_loss_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
            token, action, amount, price, sentiment, score, holder_growth, trade_id, channel, take_profit, stop_loss)
            self.db.execute_query(query, params)

            logger.info(
                f"交易执行并记录: {token} - {action} {amount} @ {price}, Channel: {channel}, Trade ID: {trade_id}")
            return {"action": action, "reason": reason, "trade_id": trade_id}

        except Exception as e:
            logger.error(f"交易执行失败: {token} - {e}")
            return None

    def get_trade_history(self, token=None, channel=None, limit=100):
        """查询交易历史"""
        try:
            query = """
            SELECT * FROM trades
            WHERE (%s IS NULL OR token = %s)
            AND (%s IS NULL OR channel = %s)
            ORDER BY timestamp DESC
            LIMIT %s
            """
            params = (token, token, channel, channel, limit)
            results = self.db.execute_query(query, params)
            if results:
                trades = [
                    {"id": r[0], "token": r[1], "action": r[2], "amount": r[3], "price": r[4],
                     "sentiment": r[5], "score": r[6], "holder_growth": r[7], "trade_id": r[8],
                     "channel": r[9], "take_profit_price": r[10], "stop_loss_price": r[11],
                     "timestamp": r[12]}
                    for r in results
                ]
                logger.info(f"查询交易历史: {len(trades)} 条记录")
                return trades
            return []
        except Exception as e:
            logger.error(f"查询交易历史失败: {e}")
            return None


if __name__ == "__main__":
    engine = TradingEngine()
    decision = engine.execute_trade("SOL", 0.85, 10, 150.0, 0.7, 0.25, "7bX8...pump")
    print(f"交易决策: {decision}")