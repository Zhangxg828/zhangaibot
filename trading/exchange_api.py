from utils.logger import setup_logger
import ccxt

logger = setup_logger("exchange_api")


class GateIOAPI:
    def __init__(self, api_key, secret):
        self.client = ccxt.gateio({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True
        })
        self.client.load_markets()
        logger.info("Gate.io API 初始化完成")

    def futures_trade(self, contract, side, size, price, leverage, take_profit=None, stop_loss=None):
        """执行Gate.io期货交易"""
        try:
            # 设置杠杆
            self.client.set_leverage(leverage, contract, params={"settle": "usdt"})

            # 下单
            order = self.client.create_order(
                symbol=contract,
                type="market",
                side=side,
                amount=size,
                price=price,
                params={"settle": "usdt", "contract": contract}
            )
            trade_id = order["id"]  # 修改变量名为 trade_id

            # 设置止盈止损
            if take_profit:
                self.client.create_order(
                    symbol=contract,
                    type="limit",
                    side="sell" if side == "buy" else "buy",
                    amount=size,
                    price=take_profit,
                    params={"settle": "usdt", "trigger_price": take_profit, "contract": contract}
                )
            if stop_loss:
                self.client.create_order(
                    symbol=contract,
                    type="limit",
                    side="sell" if side == "buy" else "buy",
                    amount=size,
                    price=stop_loss,
                    params={"settle": "usdt", "trigger_price": stop_loss, "contract": contract}
                )

            logger.info(f"Gate.io期货交易成功: {contract} - {side} {size} @ {price}, Trade ID: {trade_id}")
            return trade_id
        except Exception as e:
            logger.error(f"Gate.io期货交易失败: {e}")
            raise


if __name__ == "__main__":
    api = GateIOAPI("your_api_key", "your_secret")
    local_trade_id = api.futures_trade("BTC_USDT", "buy", 1, 50000, 5, 51000, 49000)
    print(f"Trade ID: {local_trade_id}")
