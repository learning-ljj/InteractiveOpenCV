# Memory/ExecutorMemory.py
from typing import List
from pydantic import BaseModel, Field
from Infrastructure.schema import Message

class ExecutorMemory(BaseModel):
    """执行器专用的记忆存储"""
    messages: List[Message] = Field(default_factory=list)  # 消息列表
    max_messages: int = Field(default=100)  # 最大消息数量限制

    def add_message(self, message: Message) -> None:
        """添加单条消息到记忆"""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """添加多条消息到记忆"""
        self.messages.extend(messages)
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