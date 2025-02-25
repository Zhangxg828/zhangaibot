from flask import Flask, request, render_template
import yaml

app = Flask(__name__)

def load_config():
    with open("data_pipeline/config.yaml", "r") as f:
        return yaml.safe_load(f)

def save_config(config):
    with open("data_pipeline/config.yaml", "w") as f:
        yaml.dump(config, f)

@app.route("/", methods=["GET", "POST"])
def control_panel():
    config = load_config()
    if request.method == "POST":
        config["twitter_api"]["keys"] = [
            {"consumer_key": request.form[f"twitter_key_{i}"],
             "consumer_secret": request.form[f"twitter_secret_{i}"],
             "access_token": request.form[f"twitter_token_{i}"],
             "access_token_secret": request.form[f"twitter_token_secret_{i}"]}
            for i in range(int(request.form["twitter_key_count"]))
        ]
        config["gateio_api"] = {
            "api_key": request.form["gateio_key"],
            "secret": request.form["gateio_secret"]
        }
        config["trading_params"] = {
            "leverage": int(request.form["leverage"]),
            "take_profit_percentage": float(request.form["take_profit"]),
            "stop_loss_percentage": float(request.form["stop_loss"]),
            "max_position": float(request.form["max_position"]),
            "daily_trade_limit": int(request.form["daily_trade_limit"]),
            "circuit_breaker_loss": float(request.form["circuit_breaker"])
        }
        save_config(config)
    return render_template("control_panel.html", config=config)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)