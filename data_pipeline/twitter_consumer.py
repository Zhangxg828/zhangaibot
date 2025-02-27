import tweepy
import json
from kafka import KafkaProducer
import yaml
import time
from utils.logger import setup_logger

logger = setup_logger("twitter_consumer")

class TwitterConsumer:
    def __init__(self):
        with open("data_pipeline/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.keys = self.config["twitter_api"]["keys"]
        self.current_key_idx = 0
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        self.client = None  # 在__init__中显式声明
        self.setup_client()  # 初始化client

    def setup_client(self):
        """初始化当前API Key的Twitter客户端"""
        try:
            key = self.keys[self.current_key_idx]
            self.client = tweepy.Client(
                consumer_key=key["consumer_key"],
                consumer_secret=key["consumer_secret"],
                access_token=key["access_token"],
                access_token_secret=key["access_token_secret"]
            )
            logger.info(f"使用 Twitter API Key {self.current_key_idx + 1}/{len(self.keys)}")
        except Exception as e:
            logger.error(f"初始化Twitter客户端失败: {e}")
            self.client = None  # 确保异常情况下client有定义

    def switch_key(self):
        """切换到下一个API Key"""
        self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
        self.setup_client()
        logger.info("由于流量限制，切换到下一个Twitter API Key")

    def fetch_tweets(self, query="MEME coin"):
        """获取Twitter数据并推送到Kafka"""
        try:
            if self.client is None:
                raise ValueError("Twitter客户端未初始化")
            tweets = self.client.search_recent_tweets(query=query, max_results=100)
            for tweet in tweets.data:
                data = {"text": tweet.text, "id": tweet.id, "timestamp": str(tweet.created_at)}
                self.producer.send("twitter_stream", value=data)
            logger.info(f"成功获取 {len(tweets.data)} 条推文")
        except tweepy.errors.TweepyException as e:
            if "Rate limit" in str(e):
                logger.warning("达到API流量限制，切换Key")
                self.switch_key()
                time.sleep(15)
                self.fetch_tweets(query)
            else:
                logger.error(f"Twitter API错误: {e}")
        except Exception as e:
            logger.error(f"获取推文失败: {e}")

if __name__ == "__main__":
    consumer = TwitterConsumer()
    while True:
        consumer.fetch_tweets()
        time.sleep(60)