import smtplib
from email.mime.text import MIMEText
from utils.logger import setup_logger

logger = setup_logger("alerts")

class AlertSystem:
    @staticmethod
    def send_alert(message, recipient="example@example.com"):
        """发送告警邮件"""
        try:
            msg = MIMEText(message)
            msg["Subject"] = "Crypto Trading Bot Alert"
            msg["From"] = "bot@example.com"
            msg["To"] = recipient
            with smtplib.SMTP("smtp.example.com") as server:
                server.login("username", "password")
                server.send_message(msg)
            logger.info(f"告警已发送至 {recipient}: {message}")
        except Exception as e:
            logger.error(f"发送告警失败: {e}")

if __name__ == "__main__":
    AlertSystem.send_alert("Test alert message")  # 更新调用方式