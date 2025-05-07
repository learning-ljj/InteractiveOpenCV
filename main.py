# 导入异步IO库
import asyncio

# 从项目模块导入Manus代理和日志记录器
from Agent.Manus import Manus
from Infrastructure.logger import logger


async def main():
    """主异步函数，负责初始化代理并处理用户交互"""
    
    # 创建并初始化Manus代理实例
    agent = await Manus.create()
    
    try:
        # 获取用户输入提示
        prompt = input("请输入您的请求: ")
        
        # 检查输入是否为空或仅包含空白字符
        if not prompt.strip():
            logger.warning("输入内容为空，请提供有效指令。")
            return

        # 开始处理用户请求
        logger.warning("正在处理您的请求，请稍候...")
        
        # 调用代理执行用户请求
        await agent.run(prompt)
        
        # 请求处理完成提示
        logger.info("请求处理已完成。")
        
    except KeyboardInterrupt:
        # 捕获用户中断操作(Ctrl+C)
        logger.warning("操作已被用户中断。")
        
    finally:
        # 确保代理资源被正确释放
        await agent.cleanup()


if __name__ == "__main__":
    # 程序入口点，启动异步主函数
    asyncio.run(main())
