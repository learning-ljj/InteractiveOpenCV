from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    """工具基类，定义所有工具的统一接口规范"""
    
    name: str  # 工具名称标识符，需全局唯一
    description: str  # 工具功能描述，用于LLM理解工具用途
    parameters: Optional[dict] = None  # OpenAPI格式的参数定义

    class Config:
        """Pydantic配置，允许任意类型字段"""
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """使工具实例可调用，转发到execute方法"""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """抽象方法，子类必须实现具体工具逻辑"""

    def to_param(self) -> Dict:
        """将工具转换为OpenAI函数调用格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """工具执行结果的标准封装"""
    
    output: Any = Field(default=None)  # 主输出内容
    error: Optional[str] = Field(default=None)  # 错误信息
    base64_image: Optional[str] = Field(default=None)  # 图片base64数据
    system: Optional[str] = Field(default=None)  # 系统级消息

    class Config:
        """Pydantic配置，允许任意类型字段"""
        arbitrary_types_allowed = True

    def __bool__(self):
        """布尔判断：任一字段有值返回True"""
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        """结果合并：支持输出内容拼接"""
        def combine_fields(
            field: Optional[str], 
            other_field: Optional[str], 
            concatenate: bool = True
        ):
            """字段合并逻辑：相同字段选择性拼接"""
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("无法合并工具结果")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        """字符串表示：优先显示错误信息"""
        return f"错误: {self.error}" if self.error else str(self.output)

    def replace(self, **kwargs):
        """创建新实例并替换指定字段"""
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """命令行专用的结果类型，继承基础功能"""


class ToolFailure(ToolResult):
    """表示工具执行失败的专用结果类型"""