import psycopg2
from utils.logger import setup_logger

logger = setup_logger("db")

class Database:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        logger.info("数据库连接成功")

    def insert_chain_data(self, data):
        query = """
            INSERT INTO chain_data (timestamp, token, liquidity, holder_count, tx_volume, creator_address)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.cursor.execute(query, (
            data["timestamp"], data["token"], data["liquidity"],
            data["holder_count"], data["tx_volume"], data["creator_address"]
        ))
        self.conn.commit()

    def insert_social_data(self, data):
        query = """
            INSERT INTO social_data (timestamp, source, text, sentiment_score)
            VALUES (%s, %s, %s, %s)
        """
        self.cursor.execute(query, (data["timestamp"], data["source"], data["text"], data["sentiment"]))
        self.conn.commit()

    def insert_trade_record(self, record):
        query = """
            INSERT INTO trade_records (timestamp, token, amount, price, action, reason)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.cursor.execute(query, (
            record["timestamp"], record["token"], record["amount"],
            record["price"], record["action"], record["reason"]
        ))
        self.conn.commit()

    def update_blacklist(self, address):
        query = "INSERT INTO blacklist (address) VALUES (%s) ON CONFLICT DO NOTHING"
        self.cursor.execute(query, (address,))
        self.conn.commit()

    def get_blacklist(self):
        self.cursor.execute("SELECT address FROM blacklist")
        return {row[0] for row in self.cursor.fetchall()}

    def close(self):
        self.cursor.close()
        self.conn.close()
        logger.info("数据库连接已关闭")