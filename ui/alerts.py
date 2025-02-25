from utils.logger import setup_logger

logger = setup_logger("alerts")

class AlertSystem:
    def send_alert(self, message, level="INFO"):
        """发送告警"""
        if level == "ERROR":
            logger.error(f"告警: {message}")
            # 实际部署需集成邮件或微信通知
        else:
            logger.info(f"通知: {message}")

if __name__ == "__main__":
    alert = AlertSystem()
    alert.send_alert("API调用失败", "ERROR")