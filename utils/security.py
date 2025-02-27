from cryptography.fernet import Fernet
from utils.logger import setup_logger

logger = setup_logger("security")

def generate_key():
    """生成加密密钥"""
    return Fernet.generate_key()

def encrypt_key(_enc_data, _enc_key):
    """加密数据"""
    try:
        f = Fernet(_enc_key)
        _encrypted_result = f.encrypt(_enc_data)
        logger.info("密钥加密成功")
        return _encrypted_result
    except Exception as e:
        logger.error(f"密钥加密失败: {e}")
        return None

def decrypt_key(_dec_encrypted, _dec_key):
    """解密数据"""
    try:
        f = Fernet(_dec_key)
        _decrypted_result = f.decrypt(_dec_encrypted)
        logger.info("密钥解密成功")
        return _decrypted_result
    except Exception as e:
        logger.error(f"密钥解密失败: {e}")
        return None

if __name__ == "__main__":
    key = generate_key()
    data = b"my secret data"
    encrypted = encrypt_key(data, key)
    decrypted = decrypt_key(encrypted, key)
    print(f"原始数据: {data}")
    print(f"加密数据: {encrypted}")
    print(f"解密数据: {decrypted}")