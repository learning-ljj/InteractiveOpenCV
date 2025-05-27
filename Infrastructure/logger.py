import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from Infrastructure.config import PROJECT_ROOT

_print_level = "INFO"  # 默认打印日志级别
_log_initialized = False  # 标志是否已经初始化日志

def define_log_level(
        print_level="INFO", 
        logfile_level="DEBUG", 
        run_id: str = None, 
        name: str = None):
    """配置日志级别并初始化日志记录器
    
    参数:
        print_level (str): 控制台输出日志级别
        logfile_level (str): 文件记录日志级别
        name (str): 日志文件前缀名，可选
    """
    global _print_level, _log_initialized

    if _log_initialized:
        return logger  # 如果已初始化，直接返回现有日志记录器
    
    _print_level = print_level
    _log_initialized = True  # 标记为已初始化

    # 生成日志文件名
    timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{name}_{timestamp}" if name else timestamp

    # 确保logs目录存在
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # 移除现有日志处理器
    logger.remove()
    
    # 添加控制台日志处理器
    logger.add(sys.stderr, level=print_level)
    
    # 添加文件日志处理器
    logger.add(
        log_dir / f"{log_name}.log",
        level=logfile_level,
        encoding="utf-8",
    )
    
    return logger


if __name__ == "__main__":
    # 初始化默认日志记录器
    logger = define_log_level(name="test")
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