import cmd
from utils.logger import setup_logger

logger = setup_logger("cli_interface")

class CLIInterface(cmd.Cmd):
    prompt = "TradingBot> "

    @staticmethod
    def do_score(arg):
        """查询代币评分"""
        print(f"当前代币评分: {arg}")
        logger.info(f"查询代币评分: {arg}")

    @staticmethod
    def do_exit(arg):  # noqa: arg 未使用，但为cmd.Cmd接口要求
        """退出CLI"""
        print("退出程序")
        return True

if __name__ == "__main__":
    cli = CLIInterface()
    cli.cmdloop()