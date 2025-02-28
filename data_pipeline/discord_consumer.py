import discord
from discord.ext import commands
from kafka import KafkaProducer
import yaml
from utils.logger import setup_logger

logger = setup_logger("discord_consumer")


class DiscordConsumer(commands.Bot):
    def __init__(self, config_path="data_pipeline/config.yaml"):
        """初始化Discord消费者，使用discord.py采集社群消息"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Discord Bot配置
        intents = discord.Intents.default()
        intents.message_content = True  # 启用消息内容意图
        super().__init__(command_prefix="!", intents=intents)

        # Kafka生产者
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: v.encode("utf-8"),  # 直接编码为字节
            retries=3,
            max_block_ms=5000
        )

        self.enabled = self.config["discord_api"]["enabled"]
        self.channels = self.config["discord_api"]["channels"]
        self.bot_token = self.config["discord_api"]["bot_token"]
        logger.info("Discord消费者初始化完成")

    async def on_ready(self):
        """Bot启动时的事件"""
        logger.info(f"Discord Bot已登录: {self.user.name} #{self.user.discriminator}")
        if not self.enabled:
            logger.info("Discord消费者未启用，跳过消息采集")
            await self.close()

    async def on_message(self, message):
        """处理接收到的Discord消息"""
        if not self.enabled:
            return

        # 仅处理指定频道的消息
        if str(message.channel) in self.channels and message.content:
            try:
                data = {
                    "channel": str(message.channel),
                    "text": message.content,
                    "timestamp": message.created_at.timestamp()
                }
                self.producer.send("discord_stream", value=str(data))
                logger.debug(f"推送Discord消息: {message.channel} - {message.content[:50]}")
            except Exception as e:
                logger.error(f"推送Discord消息失败: {e}")

    def run(self, token: str = None, *, bot: bool = True, reconnect: bool = True) -> None:
        """运行Discord Bot，与discord.ext.commands.Bot.run()签名一致"""
        if self.enabled:
            token_to_use = token if token else self.bot_token
            if not token_to_use:
                raise ValueError("Discord Bot Token未提供")
            try:
                super().run(token_to_use)
            except Exception as e:
                logger.error(f"Discord Bot启动失败: {e}")
        else:
            logger.info("Discord消费者已禁用")


if __name__ == "__main__":
    consumer = DiscordConsumer()
    consumer.run(consumer.bot_token)  # 直接传递bot_token
