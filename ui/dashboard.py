from flask import Flask, render_template
import ccxt
import requests
from datetime import datetime
import pytz
import ping3
from utils.logger import setup_logger
import yaml

app = Flask(__name__)
logger = setup_logger("dashboard")


def load_config():
    with open("data_pipeline/config.yaml", "r") as f:
        return yaml.safe_load(f)


def fetch_valuable_coins():
    # 模拟筛选有价值的代币（实际需从交易数据中获取）
    return [
        {"name": "BTC", "price": 50000, "exchange": "binance"},
        {"name": "ETH", "price": 3000, "exchange": "gate"}
    ]


def fetch_market_data(symbol, exchange):
    client = ccxt.binance() if exchange == "binance" else ccxt.gateio()
    ohlcv = client.fetch_ohlcv(symbol, timeframe="1h", limit=24)
    return [{"time": o[0], "open": o[1], "high": o[2], "low": o[3], "close": o[4]} for o in ohlcv]


def fetch_community_topics():
    # 模拟社区话题（实际需从Discord/Telethon获取）
    return [
        {"title": "BTC to $100K?", "source": "https://discord.com/channels/123/topic1"},
        # ...更多话题
    ]


def fetch_portfolio():
    # 模拟持仓（实际需从交易所API获取）
    return [
        {"coin": "BTC", "amount": 0.1, "value": 5000, "profit_loss": 100},
        {"coin": "ETH", "amount": 2, "value": 6000, "profit_loss": -50}
    ]


def fetch_strategy():
    # 模拟策略（实际从trading_params获取）
    return {"stop_profit": 5.0, "stop_loss": 2.0}


def fetch_balances():
    # 模拟钱包余额（实际需从交易所API获取）
    return {"gate": 10000, "binance": 5000, "total_profit_loss": 50}


def fetch_times():
    return {
        "us": datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S"),
        "beijing": datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "europe": datetime.now(pytz.timezone("Europe/London")).strftime("%Y-%m-%d %H:%M:%S")
    }


def fetch_network_status():
    return {
        "solana": ping3.ping("api.mainnet-beta.solana.com"),
        "gate": ping3.ping("api.gateio.ws"),
        "binance": ping3.ping("api.binance.com")
    }


@app.route("/dashboard")
def dashboard():
    config = load_config()
    valuable_coins = fetch_valuable_coins()
    btc_data = fetch_market_data("BTC/USDT", "binance")
    eth_data = fetch_market_data("ETH/USDT", "gate")
    fear_index = 50  # 模拟，需从LunarCrush获取
    topics = fetch_community_topics()
    portfolio = fetch_portfolio()
    strategy = fetch_strategy()
    balances = fetch_balances()
    times = fetch_times()
    network_status = fetch_network_status()

    return render_template("dashboard.html",
                           coins=valuable_coins, btc_data=btc_data, eth_data=eth_data, fear_index=fear_index,
                           topics=topics, portfolio=portfolio, strategy=strategy, balances=balances,
                           times=times, network_status=network_status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)