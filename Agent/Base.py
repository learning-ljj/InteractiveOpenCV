from abc import ABC, abstractmethod  # 抽象基类支持
from contextlib import asynccontextmanager  # 异步上下文管理器
from typing import List, Optional  # 类型注解支持

from pydantic import BaseModel, Field, model_validator  # 数据验证和设置管理

from llm import LLM  # 语言模型接口
from Infrastructure.logger import logger  # 日志记录器
from Infrastructure.sandbox.client import SANDBOX_CLIENT  # 沙箱环境客户端
from Infrastructure.schema import ROLE_TYPE, AgentState, Message  # 类型定义和数据结构

from Memory.ExecutorMemory import ExecutorMemory  # 执行代理Memory模块

class BaseAgent(BaseModel, ABC):
    """代理基类，用于管理代理状态和执行流程
    
    提供状态转换、记忆管理和基于步骤的执行循环等基础功能。
    子类必须实现`step`方法。
    """

    # 核心属性
    name: str = Field(..., description="代理的唯一名称")
    description: Optional[str] = Field(None, description="代理的可选描述信息")

    # 提示词相关
    system_prompt: Optional[str] = Field(
        None, description="系统级别的指令提示词"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="用于决定下一步行动的提示词"
    )

    # 依赖组件
    llm: LLM = Field(default_factory=LLM, description="语言模型实例")
    memory: ExecutorMemory = Field(default_factory=ExecutorMemory, description="代理的记忆存储")
    state: AgentState = Field(
        default=AgentState.IDLE, description="代理当前状态"
    )

    # 执行控制参数
    max_steps: int = Field(default=10, description="终止前的最大执行步数")
    current_step: int = Field(default=0, description="当前执行步骤计数")
    duplicate_threshold: int = 2  # 判断卡顿的重复消息阈值

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型字段
        extra = "allow"  # 允许子类添加额外字段

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """初始化代理实例，设置默认值
        
        如果llm或memory未正确初始化，则创建默认实例
        """
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, ExecutorMemory):
            self.memory = ExecutorMemory()
        return self

    @asynccontextmanager
    async def state_context(self, 
                          new_state: AgentState, 
                          should_reset: bool = True,
                          reset_on_error: bool = True):
        """代理状态上下文管理器，用于安全地切换代理状态
        
        参数说明：
            new_state (AgentState): 
                必须传入的目标状态，必须是预定义的AgentState枚举值
                示例: AgentState.RUNNING, AgentState.IDLE

            should_reset (bool, optional): 
                当任务完成时是否自动重置为初始状态（默认False）
                - True: 任务完成后将状态重置为AgentState.IDLE
                - False: 保持AgentState.FINISHED状态
                注意：若参数为True，运行run_flow：
                    当terminate将Agent状态设为finished时，
                        state_context会复原为待机状态IDLE，
                        导致flow的execute中的while循环继续运行，不会重置步数

            reset_on_error (bool, optional): 
                发生未处理异常时是否恢复原始状态（默认True）
                - True: 出现异常时恢复进入上下文前的状态
                - False: 出现异常时保持AgentState.ERROR状态

        状态处理优先级（从上到下依次判断）：
            1. 发生未捕获的异常 → 根据reset_on_error决定行为
            2. 达到FINISHED状态 → 根据should_reset决定是否重置
            3. 其他正常情况 → 自动恢复原始状态
        """
        # 参数验证
        if not isinstance(new_state, AgentState):
            raise ValueError(f"无效状态类型，期望AgentState枚举，实际收到: {type(new_state)}")

        # 保存原始状态用于可能的恢复
        original_state = self.state
        
        try:
            # 应用新状态
            self.state = new_state
            logger.debug(f"状态已切换: {original_state} → {new_state}")
            
            # 移交控制权给with块
            yield
            
            # 处理正常完成情况
            if self.state == AgentState.FINISHED:
                if should_reset:
                    self.state = AgentState.IDLE
                    logger.info("任务完成并重置为待机状态")
                else:
                    logger.info("任务完成，保持FINISHED状态")
            else:
                # 非完成状态恢复原始状态
                self.state = original_state
                logger.info(f"恢复进入前状态: {original_state}")
                
        except Exception as e:
            # 异常处理
            logger.error(f"执行过程中发生异常: {type(e).__name__}: {str(e)}")
            
            if reset_on_error:
                self.state = original_state
                logger.warning(f"已恢复原始状态: {original_state}")
            else:
                self.state = AgentState.ERROR
                logger.error("Agent保持错误状态，需要人工干预")
            
            raise  # 重新抛出异常

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """向代理记忆中添加消息
        
        参数:
            role: 消息角色类型(用户/系统/助手/工具)
            content: 消息文本内容
            base64_image: 可选，base64编码的图像数据
            **kwargs: 额外参数，主要用于工具消息
            
        功能:
            根据角色类型创建对应类型的消息对象并添加到记忆存储中
        """
        # 消息类型映射字典，将不同角色映射到对应的消息构造方法
        message_map = {
            "user": Message.user_message,  # 用户消息构造方法，用于创建用户角色消息
            "system": Message.system_message,  # 系统消息构造方法，用于创建系统角色消息 
            "assistant": Message.assistant_message,  # 助手消息构造方法，用于创建助手角色消息
            "tool": lambda content, **kw: Message.tool_message(content, **kw),  # 工具消息构造方法，使用lambda包装以支持动态参数
        }

        # 检查角色是否有效
        if role not in message_map:
            raise ValueError(f"不支持的消息角色: {role}")

        # 准备消息构造参数
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        
        # 创建消息对象并添加到记忆
        self.memory.add_message(message_map[role](content, **kwargs))

    # 新添加的方法，未完成修改
    async def plan(self) -> str:
        """设计规划代理工作流的步骤
        
        必须由子类实现以定义具体行为
        """

    # 未完成修改，需要想清楚run方法的具体实现，特别是plan模块的实现
    async def run(self, request: Optional[str] = None) -> str:
        """异步执行代理主循环
        
        处理初始请求并执行多步操作，直到完成或达到最大步数
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"无法从{self.state}状态启动代理")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"执行步骤 {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # 未完成修改
                if self.is_stuck():
                     self.handle_stuck_state()

                results.append(f"步骤 {self.current_step}: {step_result}")
            
            if self.state == AgentState.FINISHED:
                self.current_step = 0 # 重置步数
                self.memory.clear()  # 清理记忆

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"终止: 达到最大步骤数 ({self.max_steps})")
        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "未执行任何步骤"

    @abstractmethod
    async def step(self) -> str:
        """执行代理工作流中的单一步骤
        
        必须由子类实现以定义具体行为
        """


    # 未完成修改
    """原有卡顿状态的检测是检测完全相同的语句，这样会导致无法判断是否是真正意义上的卡顿
    ，卡顿状态的处理是否可以先总结出之前的流程中卡在哪些地方，然后利用LLM修改策略？"""
    def handle_stuck_state(self):
        """处理卡顿状态
        
        通过添加提示词来改变策略
        """
        stuck_prompt = "\
        检测到重复响应。请考虑新的策略，避免重复已经尝试过的无效路径。"
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"代理检测到卡顿状态。已添加提示: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """通过检测重复内容判断代理是否卡在循环中"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # 计算相同内容出现的次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property # 方法转换为只读属性，允许像访问属性一样访问方法，例如：agent.messages = [msg1, msg2]，而不是agent.messages()
    def messages(self) -> List[Message]:
        """获取代理记忆中的所有消息"""
        return self.memory.messages

    @messages.setter # 为属性创建设置器(setter)，允许通过赋值修改属性值，例如：agent.messages = [msg1, msg2]，而不是agent.messages([msg1, msg2])
    def messages(self, value: List[Message]):
        """设置代理记忆中的消息列表"""
        self.memory.messages = value
