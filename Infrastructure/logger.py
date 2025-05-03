import sys
from datetime import datetime

from loguru import logger as _logger  # 导入loguru库并重命名为_logger

from Infrastructure.config import PROJECT_ROOT  # 从项目配置中导入项目根目录


_print_level = "INFO"  # 默认打印日志级别


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """配置日志级别并初始化日志记录器
    
    参数:
        print_level (str): 控制台输出日志级别，默认为INFO
        logfile_level (str): 文件记录日志级别，默认为DEBUG
        name (str): 日志文件前缀名，可选
        
    返回:
        Logger: 配置好的日志记录器实例
    """
    global _print_level
    _print_level = print_level  # 更新全局打印级别

    # 获取当前时间并格式化为字符串
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    
    # 如果有提供名称前缀，则组合名称和时间作为日志文件名
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )

    # 移除现有日志处理器
    _logger.remove()
    
    # 添加控制台日志处理器
    _logger.add(sys.stderr, level=print_level)
    
    # 添加文件日志处理器，日志文件保存在项目logs目录下
    _logger.add(PROJECT_ROOT / f"logs/{log_name}.log", level=logfile_level)
    
    return _logger  # 返回配置好的日志记录器


# 初始化默认日志记录器
logger = define_log_level()


if __name__ == "__main__":
    # 测试日志功能
    logger.info("应用程序启动")  # 记录一般信息
    logger.debug("调试信息")  # 记录调试信息
    logger.warning("警告信息")  # 记录警告信息
    logger.error("错误信息")  # 记录错误信息
    logger.critical("严重错误信息")  # 记录严重错误信息

    # 测试异常记录
    try:
        raise ValueError("测试错误")
    except Exception as e:
        logger.exception(f"发生错误: {e}")  # 记录异常堆栈信息