from utils.logger import setup_logger

logger = setup_logger("risk_manager")

class RiskManager:
    def __init__(self, total_funds):
        self.total_funds = total_funds
        self.current_loss = 0.0
        self.positions = {}  # {token: {"amount": float, "entry_price": float}}

    def get_current_loss(self):
        """获取当前累计亏损"""
        return self.current_loss

    def check_risk(self, trade_amount, current_loss, risk_params):
        """检查交易是否符合风险限制"""
        max_position = risk_params["max_position"]
        circuit_breaker_loss = risk_params["circuit_breaker_loss"]
        total_position_value = sum(p["amount"] * p["entry_price"] for p in self.positions.values())
        if (total_position_value + trade_amount * 100) / self.total_funds * 100 > max_position:
            return False
        if current_loss > circuit_breaker_loss:
            return False
        return True

    def update_position(self, token, action, amount, price):
        """更新持仓"""
        if action == "buy":
            if token in self.positions:
                self.positions[token]["amount"] += amount
                self.positions[token]["entry_price"] = (self.positions[token]["entry_price"] + price) / 2
            else:
                self.positions[token] = {"amount": amount, "entry_price": price}
        elif action == "sell" and token in self.positions:
            self.positions[token]["amount"] -= amount
            if self.positions[token]["amount"] <= 0:
                del self.positions[token]

    def update_loss(self, loss):
        """更新累计亏损"""
        self.current_loss += loss
        logger.info(f"累计亏损更新: {self.current_loss}")