from cryptography.fernet import Fernet
from utils.logger import setup_logger

logger = setup_logger("security")

class Security:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, data):
        """加密数据"""
        encrypted = self.cipher.encrypt(data.encode())
        logger.info("数据加密完成")
        return encrypted

    def decrypt(self, encrypted_data):
        """解密数据"""
        decrypted = self.cipher.decrypt(encrypted_data).decode()
        logger.info("数据解密完成")
        return decrypted

if __name__ == "__main__":
    sec = Security()
    encrypted = sec.encrypt("my_secret_key")
    print(sec.decrypt(encrypted))