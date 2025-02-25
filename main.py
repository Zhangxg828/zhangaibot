from trading.trading_engine import TradingEngine
from utils.logger import setup_logger

logger = setup_logger("main")

def main():
    engine = TradingEngine()
    decision = engine.execute_trade("SOL", 0.85, 10, 150.0)
    while True:
        query = input("请输入问题（输入'exit'退出）: ")
        if query.lower() == "exit":
            break
        engine.interact(query, {"score": 0.85, "price": 150.0, "sentiment": 0.7, "holder_growth": 0.3, "decision": decision})

if __name__ == "__main__":
    main()