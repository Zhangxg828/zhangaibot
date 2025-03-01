import psycopg2
from psycopg2 import pool
from utils.logger import setup_logger
import yaml

logger = setup_logger("database")


class Database:
    def __init__(self, config_path="data_pipeline/config.yaml"):
        """初始化数据库连接池"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.db_config = self.config["database"]
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=10,
            dbname=self.db_config["dbname"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            host=self.db_config["host"],
            port=self.db_config["port"]
        )
        logger.info("数据库连接池初始化完成")

    def get_connection(self):
        """从连接池获取连接"""
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"获取数据库连接失败: {e}")
            return None

    def release_connection(self, conn):
        """释放连接回池"""
        self.connection_pool.putconn(conn)

    def execute_query(self, query, params=None):
        """执行SQL查询"""
        conn = self.get_connection()
        if conn is None:
            return None

        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
                if cur.description:  # 如果是SELECT查询
                    return cur.fetchall()
                return True
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            conn.rollback()
            return None
        finally:
            self.release_connection(conn)

    def close_pool(self):
        """关闭连接池"""
        self.connection_pool.closeall()
        logger.info("数据库连接池关闭")


# 示例：创建交易记录表
if __name__ == "__main__":
    db = Database()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        token VARCHAR(255),
        action VARCHAR(50),
        amount FLOAT,
        price FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    db.execute_query(create_table_query)
    logger.info("交易记录表创建完成")