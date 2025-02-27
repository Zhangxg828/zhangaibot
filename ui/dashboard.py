from utils.logger import setup_logger

logger = setup_logger("dashboard")

def start_dashboard():
    """启动Grafana看板（模拟）"""
    logger.info("Grafana看板启动中，访问: http://localhost:3000")
    # 实际部署需安装Grafana并配置数据源

if __name__ == "__main__":
    start_dashboard()