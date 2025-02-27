markdown
自动换行
复制
# Crypto Trading Bot

## 安装
1. 安装Python 3.9和Docker
2. 克隆项目: `git clone <repo_url>`
3. 安装依赖: `pip install -r requirements.txt`
4. 配置`data_pipeline/config.yaml`

## 运行
- 本地运行: `python main.py`
- Docker运行:
  ```bash
  docker build -t crypto_trading_bot .
  docker run -d -p 8000:8000 crypto_trading_bot
注意事项
确保Kafka和Grafana服务运行。
配置真实的API密钥。
text
自动换行
复制
- **作用**: 部署说明。
- **部署注意**: 根据实际环境更新内容。

---

### 部署到生产环境

1. **环境准备**:
   - Ubuntu 20.04服务器，安装Python 3.9、Docker、Kafka。
   - 配置GPU支持（CUDA 12.2）。

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
配置Kafka:
启动Kafka服务，创建solana_transactions和twitter_stream主题。
启动服务:
Twitter数据采集: python data_pipeline/twitter_consumer.py &
Kafka消费: python data_pipeline/kafka_consumer.py &
控制面板: python ui/control_panel.py &
主程序: python main.py
Docker部署:
bash
自动换行
复制
docker build -t crypto_trading_bot .
docker run -d --gpus all -p 8000:8000 -v $(pwd)/logs:/app/logs crypto_trading_bot
监控:
日志: 检查logs/目录。
控制面板: 访问http://<server_ip>:8000。
注意事项
安全性: 使用security.py加密API密钥，生产环境避免明文存储。
模型训练: 首次运行需为token_scoring.py、price_prediction.py等提供训练数据。
扩展性: 可通过修改exchange_api.py支持更多交易所。