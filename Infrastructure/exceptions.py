class ToolError(Exception):
    """工具执行过程中发生错误时抛出此异常"""

    def __init__(self, message):
        """初始化工具错误异常
        
        参数:
            message (str): 错误描述信息
        """
        self.message = message  # 存储错误信息


class OpenManusError(Exception):
    """OpenManus系统所有异常的基类"""


class TokenLimitExceeded(OpenManusError):
    """当token使用量超过限制时抛出此异常"""