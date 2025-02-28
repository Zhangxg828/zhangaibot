from flask import Flask, request, render_template
import yaml
from utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger("control_panel")

def load_config():
    """加载配置文件"""
    try:
        with open("data_pipeline/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

def save_config(config):
    """保存配置文件"""
    try:
        with open("data_pipeline/config.yaml", "w") as f:
            yaml.dump(config, f)
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")

@app.route("/", methods=["GET", "POST"])
def control_panel():
    config = load_config()
    if request.method == "POST":
        try:
            # Twitter API
            config["twitter_api"]["enabled"] = request.form.get("twitter_enabled") == "on"
            config["twitter_api"]["keys"] = [
                {"consumer_key": request.form[f"twitter_key_{i}"],
                 "consumer_secret": request.form[f"twitter_secret_{i}"],
                 "access_token": request.form[f"twitter_token_{i}"],
                 "access_token_secret": request.form[f"twitter_token_secret_{i}"]}
                for i in range(int(request.form["twitter_key_count"]))
            ]
            # Binance API
            config["binance_api"]["enabled"] = request.form.get("binance_enabled") == "on"
            config["binance_api"]["symbols"] = request.form["binance_symbols"].split(",")
            # Discord API
            config["discord_api"]["enabled"] = request.form.get("discord_enabled") == "on"
            config["discord_api"]["api_id"] = request.form["discord_api_id"]
            config["discord_api"]["api_hash"] = request.form["discord_api_hash"]
            config["discord_api"]["channels"] = request.form["discord_channels"].split(",")
            # Telegram API
            config["telegram_api"]["enabled"] = request.form.get("telegram_enabled") == "on"
            config["telegram_api"]["api_id"] = request.form["telegram_api_id"]
            config["telegram_api"]["api_hash"] = request.form["telegram_api_hash"]
            config["telegram_api"]["channels"] = request.form["telegram_channels"].split(",")
            # Glassnode API
            config["glassnode_api"]["enabled"] = request.form.get("glassnode_enabled") == "on"
            config["glassnode_api"]["api_key"] = request.form["glassnode_api_key"]
            # LunarCrush API
            config["lunarcrush_api"]["enabled"] = request.form.get("lunarcrush_enabled") == "on"
            config["lunarcrush_api"]["api_key"] = request.form["lunarcrush_api_key"]
            # Gate.io API
            config["gateio_api"]["api_key"] = request.form["gateio_key"]
            config["gateio_api"]["secret"] = request.form["gateio_secret"]
            # Trading Parameters
            config["trading_params"] = {
                "leverage": int(request.form["leverage"]),
                "take_profit_percentage": float(request.form["take_profit"]),
                "stop_loss_percentage": float(request.form["stop_loss"]),
                "max_position": float(request.form["max_position"]),
                "daily_trade_limit": int(request.form["daily_trade_limit"]),
                "circuit_breaker_loss": float(request.form["circuit_breaker"])
            }
            save_config(config)
        except Exception as e:
            logger.error(f"处理表单数据失败: {e}")
    try:
        return render_template("control_panel.html", config=config)
    except Exception as e:
        logger.error(f"加载模板文件失败: {e}")
        return "模板文件加载失败，请检查 ui/templates/control_panel.html 是否存在", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)