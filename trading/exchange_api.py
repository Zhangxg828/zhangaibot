import ccxt
from utils.logger import setup_logger

logger = setup_logger("exchange_api")

class GateIOAPI:
    def __init__(self, api_key, secret):
        self.client = ccxt.gateio({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True
        })
        self.client.load_markets()

    def get_price(self, symbol):
        """获取当前价格"""
        try:
            ticker = self.client.fetch_ticker(symbol)
            return ticker["last"]
        except Exception as e:
            logger.error(f"获取价格错误: {e}")
            return None

    def spot_trade(self, symbol, side, amount, price=None):
        """现货交易"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                type="market" if not price else "limit",
                side=side,
                amount=amount,
                price=price
            )
            logger.info(f"现货交易: {side} {symbol}, 数量: {amount}")
            return order
        except Exception as e:
            logger.error(f"现货交易错误: {e}")
            return None

    def futures_trade(self, symbol, side, amount, leverage=5, take_profit=None, stop_loss=None):
        """合约交易"""
        try:
            self.client.options["defaultType"] = "futures"
            self.client.set_leverage(leverage, symbol)
            order = self.client.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount
            )
            if take_profit:
                self.client.create_order(
                    symbol=symbol,
                    type="limit",
                    side="sell" if side == "buy" else "buy",
                    amount=amount,
                    price=take_profit
                )
            if stop_loss:
                self.client.create_order(
                    symbol=symbol,
                    type="market",  # 改为market类型
                    side="sell" if side == "buy" else "buy",
                    amount=amount,
                    params={"stopPrice": stop_loss}  # 使用params设置止损价格
                )
            logger.info(f"合约交易: {side} {symbol}, 杠杆: {leverage}")
            return order
        except Exception as e:
            logger.error(f"合约交易错误: {e}")
            return None

if __name__ == "__main__":
    api = GateIOAPI("your_key", "your_secret")
    print(api.get_price("SOL/USDT"))