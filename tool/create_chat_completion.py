from typing import Any, List, Optional, Type, Union, get_args, get_origin
from pydantic import BaseModel, Field
from tool import BaseTool

class CreateChatCompletion(BaseTool):
    """解析LLM响应，结构化llm聊天响应的生成工具，支持多种输出格式"""
    
    # 工具标识信息
    name: str = "create_chat_completion"  # 工具名称，用于系统识别
    description: str = "生成符合指定格式的结构化响应内容"  # 工具功能描述

    # 类型映射表，用于将Python类型转换为JSON Schema类型
    type_mapping: dict = {
        str: "string",    # 字符串类型
        int: "integer",   # 整型
        float: "number",  # 数字型
        bool: "boolean",  # 布尔型
        dict: "object",   # 对象类型
        list: "array",    # 数组类型
    }
    
    # 响应类型配置
    response_type: Optional[Type] = None  # 期望的响应数据类型
    required: List[str] = Field(default_factory=lambda: ["response"])  # 必填字段列表

    def __init__(self, response_type: Optional[Type] = str):
        """初始化工具实例并设置响应类型"""
        super().__init__()
        self.response_type = response_type  # 设置响应数据类型
        self.parameters = self._build_parameters()  # 根据类型构建参数结构

    def _build_parameters(self) -> dict:
        """构建符合JSON Schema规范的参数结构"""
        # 处理字符串类型响应
        if self.response_type == str:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "需要返回给用户的文本内容",
                    },
                },
                "required": self.required,
            }

        # 处理Pydantic模型类型响应
        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            schema = self.response_type.model_json_schema()
            return {
                "type": "object",
                "properties": schema["properties"],  # 直接使用模型的属性定义
                "required": schema.get("required", self.required),
            }

        # 处理其他复杂类型
        return self._create_type_schema(self.response_type)

    def _create_type_schema(self, type_hint: Type) -> dict:
        """为指定类型创建JSON Schema结构"""
        origin = get_origin(type_hint)  # 获取类型原始类
        args = get_args(type_hint)     # 获取类型参数

        # 处理基本类型（非泛型）
        if origin is None:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": self.type_mapping.get(type_hint, "string"),
                        "description": f"{type_hint.__name__}类型响应值",
                    }
                },
                "required": self.required,
            }

        # 处理列表类型
        if origin is list:
            item_type = args[0] if args else Any  # 获取列表元素类型
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": self._get_type_info(item_type),  # 定义数组元素结构
                    }
                },
                "required": self.required,
            }

        # 处理字典类型
        if origin is dict:
            value_type = args[1] if len(args) > 1 else Any  # 获取字典值类型
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "additionalProperties": self._get_type_info(value_type),
                    }
                },
                "required": self.required,
            }

        # 处理联合类型
        if origin is Union:
            return self._create_union_schema(args)

        return self._build_parameters()

    def _get_type_info(self, type_hint: Type) -> dict:
        """获取单个类型的描述信息"""
        # 处理Pydantic模型类型
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint.model_json_schema()

        # 处理基础类型
        return {
            "type": self.type_mapping.get(type_hint, "string"),
            "description": f"{getattr(type_hint, '__name__', '任意')}类型值",
        }

    def _create_union_schema(self, types: tuple) -> dict:
        """创建联合类型的参数结构"""
        return {
            "type": "object",
            "properties": {
                "response": {"anyOf": [self._get_type_info(t) for t in types]}  # 支持多种可能类型
            },
            "required": self.required,
        }

    async def execute(self, required: list | None = None, **kwargs) -> Any:
        """执行工具调用并返回格式化响应
        
        参数:
            required: 必填字段列表，默认为类定义
            **kwargs: 响应数据字段
            
        返回:
            根据response_type转换后的结果
        """
        required = required or self.required  # 使用默认必填字段

        # 处理多个必填字段的情况
        if isinstance(required, list) and len(required) > 0:
            if len(required) == 1:
                required_field = required[0]
                result = kwargs.get(required_field, "")  # 获取单个字段值
            else:
                # 返回包含所有必填字段的字典
                return {field: kwargs.get(field, "") for field in required}
        else:
            required_field = "response"
            result = kwargs.get(required_field, "")

        # 类型转换处理
        if self.response_type == str:
            return result  # 直接返回字符串

        # 处理Pydantic模型类型
        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            return self.response_type(**kwargs)  # 转换为模型实例

        # 处理容器类型（假设数据已正确格式化）
        if get_origin(self.response_type) in (list, dict):
            return result

        # 尝试类型转换
        try:
            return self.response_type(result)
        except (ValueError, TypeError):
            return result  # 转换失败时返回原始值