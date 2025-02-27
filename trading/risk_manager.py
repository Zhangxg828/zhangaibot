from utils.logger import setup_logger

logger = setup_logger("risk_manager")

class RiskManager:
    def __init__(self, total_funds):
        self.total_funds = total_funds
        self.daily_loss = 0

    def check_risk(self, trade_amount, current_loss, risk_params):
        """检查交易风险"""
        if trade_amount > self.total_funds * risk_params["max_position"] / 100:
            logger.warning("交易金额超过最大持仓限制")
            return False
        if self.daily_loss + current_loss >= self.total_funds * risk_params["circuit_breaker_loss"] / 100:
            logger.error("触发熔断机制，停止交易")
            return False
        return True

    def update_loss(self, loss):
        """更新每日亏损"""
        self.daily_loss += loss
        logger.info(f"当前每日亏损: {self.daily_loss}")

if __name__ == "__main__":
    manager = RiskManager(1000)
    params = {"max_position": 20, "circuit_breaker_loss": 10}
    print(manager.check_risk(300, 50, params))  # False