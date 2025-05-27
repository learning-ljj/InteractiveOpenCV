import asyncio
import time
from datetime import datetime

from Agent.Manus import Manus
from flow.flow_factory import FlowFactory, FlowType
from Infrastructure.logger import logger, define_log_level

async def run_flow():
    """主异步函数，负责运行规划流程"""
    
    # 初始化代理字典，目前只包含Manus代理
    agents = {
        "manus": Manus(),  # Manus代理实例
    }

    try:
        # 获取用户输入提示
        prompt = input("请输入您的请求: ")

        # 检查输入是否为空或仅包含空白字符
        if prompt.strip().isspace() or not prompt:
            logger.warning("输入内容为空，请提供有效指令。")
            return

        # 创建规划流程实例
        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,  # 使用规划流程类型
            agents=agents,  # 传入代理字典
        )
        logger.warning("正在处理您的请求，请稍候...")

        try:
            # 记录开始时间
            start_time = time.time()
            
            # 执行流程，设置60分钟超时
            result = await asyncio.wait_for(
                flow.execute(prompt),  # 执行流程
                timeout=3600,  # 60分钟超时限制
            )
            
            # 计算并记录处理时间
            elapsed_time = time.time() - start_time
            logger.info(f"请求处理完成，耗时: {elapsed_time:.2f} 秒")
            logger.info(result)  # 输出处理结果
            
        except asyncio.TimeoutError:
            # 处理超时情况
            logger.error("请求处理超时(超过1小时)")
            logger.info("操作因超时终止，请尝试更简单的请求。")

    except KeyboardInterrupt:
        # 处理用户中断操作(Ctrl+C)
        logger.info("操作已被用户取消。")
    except Exception as e:
        # 处理其他异常
        logger.error(f"发生错误: {str(e)}")


if __name__ == "__main__":
    # 生成单次运行的唯一ID（时间戳）
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 初始化默认日志记录器
    logger = define_log_level(run_id=RUN_ID)
    # 运行主异步函数
    asyncio.run(run_flow())
