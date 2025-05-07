from enum import Enum
from typing import Dict, List, Union

from Agent.Base import BaseAgent
from flow.base import BaseFlow
from flow.planning import PlanningFlow


class FlowType(str, Enum):
    """流程类型枚举"""
    PLANNING = "planning"  # 规划型流程


class FlowFactory:
    """流程工厂类，用于创建不同类型的多代理流程"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        """创建指定类型的流程
        
        参数:
            flow_type: 流程类型枚举值
            agents: 可接受单个代理、代理列表或代理字典
            **kwargs: 其他传递给流程构造函数的参数
            
        返回:
            初始化后的流程实例
            
        异常:
            ValueError: 当传入不支持的流程类型时抛出
        """
        # 流程类型与实现类的映射关系
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }

        # 获取对应的流程类
        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"不支持的流程类型: {flow_type}")

        # 创建并返回流程实例
        return flow_class(agents, **kwargs)
