# 导入必要的模块
from enum import Enum  # 枚举类型支持
from typing import Any, List, Literal, Optional, Union  # 类型提示相关

from pydantic import BaseModel, Field  # 数据验证和设置管理


class Role(str, Enum):
    """定义消息角色枚举类型"""
    SYSTEM = "system"     # 系统角色
    USER = "user"         # 用户角色
    ASSISTANT = "assistant"  # 助手角色
    TOOL = "tool"         # 工具角色


# 获取所有角色值的元组
ROLE_VALUES = tuple(role.value for role in Role)
# 创建字面量类型用于类型检查
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """定义工具选择选项枚举类型"""
    NONE = "none"      # 不使用工具
    AUTO = "auto"      # 自动选择工具
    REQUIRED = "required"  # 必须使用工具


# 获取所有工具选择值的元组
TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
# 创建字面量类型用于类型检查
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """定义代理执行状态枚举类型"""
    IDLE = "IDLE"      # 空闲状态
    RUNNING = "RUNNING"  # 运行中状态
    FINISHED = "FINISHED"  # 已完成状态
    ERROR = "ERROR"    # 错误状态


class Function(BaseModel):
    """定义函数调用结构"""
    name: str          # 函数名称
    arguments: str     # 函数参数(JSON字符串格式)


class ToolCall(BaseModel):
    """表示消息中的工具/函数调用"""
    id: str            # 调用ID
    type: str = "function"  # 调用类型，默认为"function"
    function: Function  # 函数调用详情


class Message(BaseModel):
    """表示对话中的聊天消息"""
    role: ROLE_TYPE = Field(...)  # 消息角色，必需字段
    content: Optional[str] = Field(default=None)  # 消息内容，可选
    tool_calls: Optional[List[ToolCall]] = Field(default=None)  # 工具调用列表，可选
    name: Optional[str] = Field(default=None)  # 名称标识，可选
    tool_call_id: Optional[str] = Field(default=None)  # 工具调用ID，可选
    base64_image: Optional[str] = Field(default=None)  # base64编码图像，可选

    def __add__(self, other) -> List["Message"]:
        """重载加法运算符，支持Message与列表或Message相加"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"不支持的操作数类型: '{type(self).__name__}' 和 '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """重载右加法运算符，支持列表与Message相加"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"不支持的操作数类型: '{type(other).__name__}' 和 '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """将消息转换为字典格式"""
        message = {"role": self.role}  # 基础消息结构
        if self.content is not None:
            message["content"] = self.content  # 添加内容
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]  # 添加工具调用
        if self.name is not None:
            message["name"] = self.name  # 添加名称
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id  # 添加工具调用ID
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image  # 添加base64图像
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建用户消息"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """创建系统消息"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """创建助手消息"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建工具消息"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """从原始工具调用创建ToolCallsMessage
        
        参数:
            tool_calls: 来自LLM的原始工具调用
            content: 可选的消息内容
            base64_image: 可选的base64编码图像
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    """定义对话记忆存储"""
    messages: List[Message] = Field(default_factory=list)  # 消息列表
    max_messages: int = Field(default=100)  # 最大消息数量限制

    def add_message(self, message: Message) -> None:
        """添加单条消息到记忆"""
        self.messages.append(message)
        # 可选: 实现消息数量限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """添加多条消息到记忆"""
        self.messages.extend(messages)
        # 可选: 实现消息数量限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """清除所有消息"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """获取最近的n条消息"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """将消息列表转换为字典列表"""
        return [msg.to_dict() for msg in self.messages]
