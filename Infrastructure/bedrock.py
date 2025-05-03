import json
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

import boto3  # AWS SDK for Python


# 全局变量，用于在函数调用间跟踪当前工具使用ID
# 临时解决方案
CURRENT_TOOLUSE_ID = None


class OpenAIResponse:
    """处理OpenAI风格响应格式的类"""
    
    def __init__(self, data):
        """初始化响应对象，递归转换嵌套字典和列表
        
        参数:
            data (dict): 要转换的原始数据
        """
        for key, value in data.items():
            if isinstance(value, dict):
                value = OpenAIResponse(value)  # 递归转换字典
            elif isinstance(value, list):
                value = [
                    OpenAIResponse(item) if isinstance(item, dict) else item
                    for item in value
                ]  # 递归转换列表中的字典
            setattr(self, key, value)  # 设置对象属性

    def model_dump(self, *args, **kwargs):
        """将对象转换为字典并添加时间戳
        
        返回:
            dict: 包含所有属性和创建时间的字典
        """
        data = self.__dict__
        data["created_at"] = datetime.now().isoformat()  # 添加ISO格式时间戳
        return data


class BedrockClient:
    """与Amazon Bedrock交互的主客户端类"""
    
    def __init__(self):
        """初始化Bedrock客户端，需要先配置AWS环境"""
        try:
            self.client = boto3.client("bedrock-runtime")  # 创建Bedrock运行时客户端
            self.chat = Chat(self.client)  # 初始化聊天接口
        except Exception as e:
            print(f"初始化Bedrock客户端错误: {e}")
            sys.exit(1)  # 初始化失败退出程序


class Chat:
    """聊天接口类"""
    
    def __init__(self, client):
        """初始化聊天接口
        
        参数:
            client: Bedrock客户端实例
        """
        self.completions = ChatCompletions(client)  # 初始化聊天补全功能


class ChatCompletions:
    """处理聊天补全功能的核心类"""
    
    def __init__(self, client):
        """初始化聊天补全功能
        
        参数:
            client: Bedrock客户端实例
        """
        self.client = client

    def _convert_openai_tools_to_bedrock_format(self, tools):
        """将OpenAI函数调用格式转换为Bedrock工具格式
        
        参数:
            tools (list): OpenAI格式的工具列表
            
        返回:
            list: Bedrock格式的工具列表
        """
        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                bedrock_tool = {
                    "toolSpec": {
                        "name": function.get("name", ""),  # 工具名称
                        "description": function.get("description", ""),  # 工具描述
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": function.get("parameters", {}).get(
                                    "properties", {}
                                ),  # 参数属性
                                "required": function.get("parameters", {}).get(
                                    "required", []
                                ),  # 必填参数
                            }
                        },
                    }
                }
                bedrock_tools.append(bedrock_tool)
        return bedrock_tools

    def _convert_openai_messages_to_bedrock_format(self, messages):
        """将OpenAI消息格式转换为Bedrock消息格式
        
        参数:
            messages (list): OpenAI格式的消息列表
            
        返回:
            tuple: (系统提示, Bedrock格式的消息列表)
        """
        bedrock_messages = []
        system_prompt = []
        for message in messages:
            if message.get("role") == "system":
                system_prompt = [{"text": message.get("content")}]
            elif message.get("role") == "user":
                bedrock_message = {
                    "role": message.get("role", "user"),
                    "content": [{"text": message.get("content")}],
                }
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "assistant":
                bedrock_message = {
                    "role": "assistant",
                    "content": [{"text": message.get("content")}],
                }
                openai_tool_calls = message.get("tool_calls", [])
                if openai_tool_calls:
                    bedrock_tool_use = {
                        "toolUseId": openai_tool_calls[0]["id"],
                        "name": openai_tool_calls[0]["function"]["name"],
                        "input": json.loads(
                            openai_tool_calls[0]["function"]["arguments"]
                        ),
                    }
                    bedrock_message["content"].append({"toolUse": bedrock_tool_use})
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = openai_tool_calls[0]["id"]
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "tool":
                bedrock_message = {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": CURRENT_TOOLUSE_ID,
                                "content": [{"text": message.get("content")}],
                            }
                        }
                    ],
                }
                bedrock_messages.append(bedrock_message)
            else:
                raise ValueError(f"无效角色: {message.get('role')}")
        return system_prompt, bedrock_messages

    def _convert_bedrock_response_to_openai_format(self, bedrock_response):
        """将Bedrock响应格式转换为OpenAI格式
        
        参数:
            bedrock_response (dict): Bedrock原始响应
            
        返回:
            OpenAIResponse: OpenAI格式的响应对象
        """
        content = ""
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            content_array = bedrock_response["output"]["message"]["content"]
            content = "".join(item.get("text", "") for item in content_array)
        if content == "":
            content = "."

        # 处理响应中的工具调用
        openai_tool_calls = []
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            for content_item in bedrock_response["output"]["message"]["content"]:
                if content_item.get("toolUse"):
                    bedrock_tool_use = content_item["toolUse"]
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = bedrock_tool_use["toolUseId"]
                    openai_tool_call = {
                        "id": CURRENT_TOOLUSE_ID,
                        "type": "function",
                        "function": {
                            "name": bedrock_tool_use["name"],
                            "arguments": json.dumps(bedrock_tool_use["input"]),
                        },
                    }
                    openai_tool_calls.append(openai_tool_call)

        # 构建最终的OpenAI格式响应
        openai_format = {
            "id": f"chatcmpl-{uuid.uuid4()}",  # 生成唯一ID
            "created": int(time.time()),  # 当前时间戳
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [
                {
                    "finish_reason": bedrock_response.get("stopReason", "end_turn"),
                    "index": 0,
                    "message": {
                        "content": content,
                        "role": bedrock_response.get("output", {})
                        .get("message", {})
                        .get("role", "assistant"),
                        "tool_calls": openai_tool_calls if openai_tool_calls != [] else None,
                        "function_call": None,
                    },
                }
            ],
            "usage": {
                "completion_tokens": bedrock_response.get("usage", {}).get("outputTokens", 0),
                "prompt_tokens": bedrock_response.get("usage", {}).get("inputTokens", 0),
                "total_tokens": bedrock_response.get("usage", {}).get("totalTokens", 0),
            },
        }
        return OpenAIResponse(openai_format)

    async def _invoke_bedrock(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        """非流式调用Bedrock模型
        
        参数:
            model (str): 模型ID
            messages (list): 消息列表
            max_tokens (int): 最大token数
            temperature (float): 温度参数
            tools (list): 工具列表
            tool_choice: 工具选择策略
            
        返回:
            OpenAIResponse: OpenAI格式的响应
        """
        system_prompt, bedrock_messages = self._convert_openai_messages_to_bedrock_format(messages)
        response = self.client.converse(
            modelId=model,
            system=system_prompt,
            messages=bedrock_messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            toolConfig={"tools": tools} if tools else None,
        )
        openai_response = self._convert_bedrock_response_to_openai_format(response)
        return openai_response

    async def _invoke_bedrock_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        """流式调用Bedrock模型
        
        参数:
            model (str): 模型ID
            messages (list): 消息列表
            max_tokens (int): 最大token数
            temperature (float): 温度参数
            tools (list): 工具列表
            tool_choice: 工具选择策略
            
        返回:
            OpenAIResponse: OpenAI格式的响应
        """
        system_prompt, bedrock_messages = self._convert_openai_messages_to_bedrock_format(messages)
        response = self.client.converse_stream(
            modelId=model,
            system=system_prompt,
            messages=bedrock_messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            toolConfig={"tools": tools} if tools else None,
        )

        # 初始化响应结构
        bedrock_response = {
            "output": {"message": {"role": "", "content": []}},
            "stopReason": "",
            "usage": {},
            "metrics": {},
        }
        bedrock_response_text = ""
        bedrock_response_tool_input = ""

        # 处理流式响应
        stream = response.get("stream")
        if stream:
            for event in stream:
                if event.get("messageStart", {}).get("role"):
                    bedrock_response["output"]["message"]["role"] = event["messageStart"]["role"]
                if event.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                    bedrock_response_text += event["contentBlockDelta"]["delta"]["text"]
                    print(event["contentBlockDelta"]["delta"]["text"], end="", flush=True)
                if event.get("contentBlockStop", {}).get("contentBlockIndex") == 0:
                    bedrock_response["output"]["message"]["content"].append(
                        {"text": bedrock_response_text}
                    )
                if event.get("contentBlockStart", {}).get("start", {}).get("toolUse"):
                    bedrock_tool_use = event["contentBlockStart"]["start"]["toolUse"]
                    tool_use = {
                        "toolUseId": bedrock_tool_use["toolUseId"],
                        "name": bedrock_tool_use["name"],
                    }
                    bedrock_response["output"]["message"]["content"].append(
                        {"toolUse": tool_use}
                    )
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = bedrock_tool_use["toolUseId"]
                if event.get("contentBlockDelta", {}).get("delta", {}).get("toolUse"):
                    bedrock_response_tool_input += event["contentBlockDelta"]["delta"]["toolUse"]["input"]
                    print(event["contentBlockDelta"]["delta"]["toolUse"]["input"], end="", flush=True)
                if event.get("contentBlockStop", {}).get("contentBlockIndex") == 1:
                    bedrock_response["output"]["message"]["content"][1]["toolUse"]["input"] = json.loads(
                        bedrock_response_tool_input
                    )
        print()
        openai_response = self._convert_bedrock_response_to_openai_format(bedrock_response)
        return openai_response

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Optional[bool] = True,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        """聊天补全的主入口点
        
        参数:
            model (str): 模型ID
            messages (list): 消息列表
            max_tokens (int): 最大token数
            temperature (float): 温度参数
            stream (bool): 是否使用流式响应
            tools (list): 工具列表
            tool_choice: 工具选择策略
            
        返回:
            OpenAIResponse: OpenAI格式的响应
        """
        bedrock_tools = []
        if tools is not None:
            bedrock_tools = self._convert_openai_tools_to_bedrock_format(tools)
        if stream:
            return self._invoke_bedrock_stream(
                model,
                messages,
                max_tokens,
                temperature,
                bedrock_tools,
                tool_choice,
                **kwargs,
            )
        else:
            return self._invoke_bedrock(
                model,
                messages,
                max_tokens,
                temperature,
                bedrock_tools,
                tool_choice,
                **kwargs,
            )
