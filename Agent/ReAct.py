from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from Agent.Base import BaseAgent  # 从基础模块导入代理基类
from llm import LLM  # 导入语言模型接口
from Infrastructure.schema import AgentState  # 导入代理状态和记忆相关类型
from Memory.ExecutorMemory import ExecutorMemory  # 执行代理Memory模块


class ReActAgent(BaseAgent, ABC):
    """ReAct代理框架，继承自Agent/Base并实现思考-执行循环模式"""
    
    name: str  # 代理名称，必须由子类指定
    description: Optional[str] = None  # 代理功能描述，可选

    # 提示词相关配置
    system_prompt: Optional[str] = None  # 系统级提示词，用于初始化代理行为
    next_step_prompt: Optional[str] = None  # 下一步行动决策提示词

    # 核心组件
    llm: Optional[LLM] = Field(default_factory=LLM)  # 语言模型实例，默认自动创建
    memory: ExecutorMemory = Field(default_factory=ExecutorMemory)  # 记忆存储实例，默认自动创建
    state: AgentState = AgentState.IDLE  # 代理初始状态设为空闲

    # 执行控制参数
    max_steps: int = 10  # 最大执行步数限制
    current_step: int = 0  # 当前已执行步数计数器

    # 未完成修改,需要想清楚think到底需要完成什么任务，而不是仅仅返回一个bool值
    @abstractmethod
    async def think(self) -> bool:
        """处理当前状态并决定下一步行动
        
        返回:
            bool: 是否需要执行行动(True)或只需思考(False)
        """

    @abstractmethod
    async def act(self) -> str:
        """执行已决定的行动
        
        返回:
            str: 行动执行结果描述
        """

    # 未完成修改，需要想清楚如何实现step方法
    async def step(self) -> str:
        """执行单步操作：思考并执行
        
        返回:
            str: 执行结果或思考完成信息
        """
        should_act = await self.think()  # 先进行思考决策
        if not should_act:
            return "思考完成 - 无需执行行动"  # 当think返回False时跳过执行
        return await self.act()  # 执行具体行动并返回结果