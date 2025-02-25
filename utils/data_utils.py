import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("data_utils")

def normalize_data(df):
    """归一化数据"""
    try:
        normalized = (df - df.min()) / (df.max() - df.min())
        logger.info("数据归一化完成")
        return normalized
    except Exception as e:
        logger.error(f"数据归一化错误: {e}")
        return df

if __name__ == "__main__":
    df = pd.DataFrame({"price": [100, 200, 300]})
    print(normalize_data(df))