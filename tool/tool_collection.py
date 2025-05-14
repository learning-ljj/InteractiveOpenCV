"""工具集合类，用于统一管理多个工具实例"""
from typing import Any, Dict, List

from Infrastructure.exceptions import ToolError
from Infrastructure.logger import logger
from tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """工具集合类，提供工具的统一管理和执行功能"""

    class Config:
        """Pydantic配置类，允许任意类型"""
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        """初始化工具集合
        Args:
            *tools: 可变参数，可传入多个工具实例
        """
        self.tools = tools  # 工具元组，保存所有工具
        self.tool_map = {tool.name: tool for tool in tools}  # 工具名称到实例的映射字典

    def __iter__(self):
        """实现迭代器协议，支持直接遍历工具集合"""
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        """将工具集合转换为OpenAPI格式的参数列表
        Returns:
            符合OpenAPI规范的工具参数列表
        """
        return [tool.to_param() for tool in self.tools]

    async def execute(self, *, name: str, tool_input: Dict[str, Any] = None) -> ToolResult:
        """执行指定名称的工具
        Args:
            name: 工具名称
            tool_input: 工具输入参数字典
        Returns:
            ToolResult: 工具执行结果对象
        """
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"无效的工具名称: {name}")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """顺序执行集合中的所有工具
        Returns:
            所有工具的执行结果列表
        """
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        """根据名称获取工具实例
        Args:
            name: 工具名称
        Returns:
            对应的工具实例，不存在则返回None
        """
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        """向集合中添加单个工具
        Args:
            tool: 要添加的工具实例
        Returns:
            self: 返回当前实例以支持链式调用
        """
        if tool.name in self.tool_map:
            logger.warning(f"工具 {tool.name} 已存在，跳过添加")
            return self

        self.tools += (tool,)  # 添加到工具元组
        self.tool_map[tool.name] = tool  # 更新映射字典
        return self

    def add_tools(self, *tools: BaseTool):
        """批量添加多个工具到集合中
        Args:
            *tools: 可变参数，可传入多个工具实例
        Returns:
            self: 返回当前实例以支持链式调用
        """
        for tool in tools:
            self.add_tool(tool)
        return self
