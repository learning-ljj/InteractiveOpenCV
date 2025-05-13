from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from Agent.Base import BaseAgent


class BaseFlow(BaseModel, ABC):
    """流程基类，支持多代理协同执行的抽象基础"""

    agents: Dict[str, BaseAgent]  # 代理字典，键为代理名称，值为代理实例
    tools: Optional[List] = None  # 可选工具列表
    primary_agent_key: Optional[str] = None  # 主代理的键名

    class Config:
        """Pydantic配置类"""
        arbitrary_types_allowed = True  # 允许任意类型

    def __init__(
        self, 
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], 
        **data
    ):
        """初始化流程实例
        
        参数:
            agents: 可接受单个代理、代理列表或代理字典
            **data: 其他初始化数据
        """
        # 处理不同类型的代理输入
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}  # 单个代理转为默认字典
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}  # 列表转为带编号的字典
        else:
            agents_dict = agents  # 直接使用字典

        # 如果未指定主代理，使用第一个代理作为主代理
        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict: # 若 未指定主代理 且 存在代理字典，注意优先级
            primary_key = next(iter(agents_dict))
            data["primary_agent_key"] = primary_key

        # 设置代理字典
        data["agents"] = agents_dict

        # 调用父类初始化
        super().__init__(**data)

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """获取当前的主代理实例"""
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """根据键名获取指定代理
        
        参数:
            key: 代理键名
            
        返回:
            对应的代理实例，如果不存在则返回None
        """
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """向流程中添加新代理
        
        参数:
            key: 代理键名
            agent: 代理实例
        """
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """执行流程的核心抽象方法
        
        参数:
            input_text: 输入文本
            
        返回:
            执行结果文本
        """