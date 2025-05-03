import math
from typing import Dict, List, Optional, Union

import tiktoken  # 用于计算token数量的库
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (  # 重试库，用于API调用失败时自动重试
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from Infrastructure.bedrock import BedrockClient  # AWS Bedrock客户端
from Infrastructure.config import LLMSettings, config  # 配置相关
from Infrastructure.exceptions import TokenLimitExceeded  # 自定义异常
from Infrastructure.logger import logger  # 日志记录器
from Infrastructure.schema import (  # 数据模型和类型定义
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

# 支持推理的特殊模型列表
REASONING_MODELS = ["o1", "o3-mini"]
# 支持多模态(图像)的模型列表
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    """用于计算消息token数量的工具类"""
    
    # Token计算相关常量
    BASE_MESSAGE_TOKENS = 4  # 每条消息的基础token数
    FORMAT_TOKENS = 2  # 消息格式的额外token数
    LOW_DETAIL_IMAGE_TOKENS = 85  # 低细节图像的固定token数
    HIGH_DETAIL_TILE_TOKENS = 170  # 高细节图像每个tile的token数

    # 图像处理相关常量
    MAX_SIZE = 2048  # 图像最大尺寸
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768  # 高细节图像的短边目标尺寸
    TILE_SIZE = 512  # 图像分块尺寸

    def __init__(self, tokenizer):
        """初始化token计数器"""
        self.tokenizer = tokenizer  # tokenizer实例

    def count_text(self, text: str) -> int:
        """计算文本字符串的token数量"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        根据细节级别和尺寸计算图像的token数量
        
        对于"low"细节: 固定85 tokens
        对于"high"细节:
        1. 缩放到2048x2048正方形内
        2. 将最短边缩放到768px
        3. 计算512px的tile数量(每个170 tokens)
        4. 添加85 tokens
        """
        detail = image_item.get("detail", "medium")  # 默认为中等细节

        # 低细节图像直接返回固定token数
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # 中等细节图像使用高细节计算方式(OpenAI没有单独的中等细节计算方式)
        if detail == "high" or detail == "medium":
            # 如果提供了图像尺寸
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # 当无法获取尺寸或细节级别未知时的默认值
        if detail == "high":
            # 默认使用1024x1024图像计算高细节token数
            return self._calculate_high_detail_tokens(1024, 1024)  # 765 tokens
        elif detail == "medium":
            # 中等细节图像的默认token数
            return 1024
        else:
            # 未知细节级别时使用中等细节
            return 1024

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """根据尺寸计算高细节图像的token数量"""
        # 步骤1: 缩放到MAX_SIZE x MAX_SIZE正方形内
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # 步骤2: 缩放使最短边达到HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # 步骤3: 计算512px的tile数量
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # 步骤4: 计算最终token数量
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """计算消息内容的token数量"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """计算工具调用的token数量"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算消息列表中所有消息的总token数量"""
        total_tokens = self.FORMAT_TOKENS  # 基础格式token

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # 每条消息的基础token

            # 添加角色token
            tokens += self.count_text(message.get("role", ""))

            # 添加内容token
            if "content" in message:
                tokens += self.count_content(message["content"])

            # 添加工具调用token
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # 添加名称和tool_call_id的token
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    """LLM客户端类，封装了与OpenAI/Claude等LLM的交互逻辑"""
    
    _instances: Dict[str, "LLM"] = {}  # 单例模式存储实例

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        """单例模式实现"""
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        """初始化LLM客户端"""
        if not hasattr(self, "client"):  # 避免重复初始化
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model  # 模型名称
            self.max_tokens = llm_config.max_tokens  # 最大生成token数
            self.temperature = llm_config.temperature  # 温度参数
            self.api_type = llm_config.api_type  # API类型(openai/azure/aws)
            self.api_key = llm_config.api_key  # API密钥
            self.api_version = llm_config.api_version  # API版本(azure专用)
            self.base_url = llm_config.base_url  # API基础URL

            # Token计数相关属性
            self.total_input_tokens = 0  # 累计输入token数
            self.total_completion_tokens = 0  # 累计生成token数
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )  # 最大输入token限制

            # 初始化tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # 如果模型不在tiktoken预设中，使用cl100k_base作为默认
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            # 根据API类型初始化不同的客户端
            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()  # AWS Bedrock客户端
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)  # token计数器实例

    def count_tokens(self, text: str) -> int:
        """计算文本中的token数量"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的token数量"""
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """更新token计数"""
        # 只有在设置了max_input_tokens时才跟踪token
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """检查是否超过token限制"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # 如果没有设置max_input_tokens，总是返回True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """生成token限制超出的错误消息"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        将消息格式化为OpenAI消息格式
        
        参数:
            messages: 可以是dict或Message对象的消息列表
            supports_images: 目标模型是否支持图像输入
            
        返回:
            List[dict]: 格式化后的OpenAI格式消息列表
            
        异常:
            ValueError: 如果消息无效或缺少必填字段
            TypeError: 如果提供了不支持的消息类型
        """
        formatted_messages = []

        for message in messages:
            # 将Message对象转换为字典
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # 确保字典消息包含必填字段
                if "role" not in message:
                    raise ValueError("消息字典必须包含'role'字段")

                # 处理base64图像(如果模型支持图像)
                if supports_images and message.get("base64_image"):
                    # 初始化或转换内容为适当格式
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # 将字符串项转换为适当的文本对象
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # 添加图像到内容
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # 移除base64_image字段
                    del message["base64_image"]
                # 如果模型不支持图像但消息有base64_image，优雅处理
                elif not supports_images and message.get("base64_image"):
                    # 只移除base64_image字段并保留文本内容
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: 不包含该消息
            else:
                raise TypeError(f"不支持的消息类型: {type(message)}")

        # 验证所有消息都有必填字段
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"无效的角色: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),  # 随机指数等待1-60秒
        stop=stop_after_attempt(6),  # 最多重试6次
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # 不重试TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        向LLM发送提示并获取响应
        
        参数:
            messages: 对话消息列表
            system_msgs: 可选的系统消息前缀
            stream: 是否流式传输响应
            temperature: 响应的采样温度
            
        返回:
            str: 生成的响应
            
        异常:
            TokenLimitExceeded: 如果超出token限制
            ValueError: 如果消息无效或响应为空
            OpenAIError: API调用失败后重试
            Exception: 意外错误
        """
        try:
            # 检查模型是否支持图像
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化系统消息和用户消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入token数量
            input_tokens = self.count_message_tokens(messages)

            # 检查是否超出token限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 抛出不会被重试的特殊异常
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            # 设置模型特定参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # 非流式请求
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("从LLM获取的响应为空或无效")

                # 更新token计数
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # 流式请求，在发送请求前更新估计的token计数
            self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)

            print()  # 流式传输后换行
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("从流式LLM获取的响应为空")

            # 估计流式响应的完成token数
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"流式响应估计的完成token数: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # 重新抛出token限制错误而不记录日志
            raise
        except ValueError:
            logger.exception(f"验证错误")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API错误")
            if isinstance(oe, AuthenticationError):
                logger.error("认证失败，请检查API密钥")
            elif isinstance(oe, RateLimitError):
                logger.error("达到速率限制，考虑增加重试次数")
            elif isinstance(oe, APIError):
                logger.error(f"API错误: {oe}")
            raise
        except Exception:
            logger.exception(f"ask方法中的意外错误")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        向LLM发送带有图像的提示并获取响应
        
        参数:
            messages: 对话消息列表
            images: 图像URL或图像数据字典列表
            system_msgs: 可选的系统消息前缀
            stream: 是否流式传输响应
            temperature: 响应的采样温度
            
        返回:
            str: 生成的响应
            
        异常:
            TokenLimitExceeded: 如果超出token限制
            ValueError: 如果消息无效或响应为空
            OpenAIError: API调用失败后重试
            Exception: 意外错误
        """
        try:
            # 对于ask_with_images，我们总是设置supports_images为True
            # 因为此方法应该只被支持图像的模型调用
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"模型 {self.model} 不支持图像。请使用以下模型之一: {MULTIMODAL_MODELS}"
                )

            # 格式化消息，启用图像支持
            formatted_messages = self.format_messages(messages, supports_images=True)

            # 确保最后一条消息来自用户，以便附加图像
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "最后一条消息必须来自用户才能附加图像"
                )

            # 处理最后一条用户消息以包含图像
            last_message = formatted_messages[-1]

            # 如果需要，将内容转换为多模态格式
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # 添加图像到内容
            for image in images:
                if isinstance(image, str):
                    # 处理字符串形式的图像URL
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    # 处理包含URL字段的字典
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    # 处理已经是正确格式的图像字典
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"不支持的图像格式: {image}")

            # 使用多模态内容更新消息
            last_message["content"] = multimodal_content

            # 如果提供了系统消息，添加到消息列表开头
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # 计算token并检查限制
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # 设置API参数
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # 添加模型特定参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # 处理非流式请求
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("从LLM获取的响应为空或无效")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # 处理流式请求
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # 流式传输后换行
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("从流式LLM获取的响应为空")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"ask_with_images中的验证错误: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API错误: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("认证失败，请检查API密钥")
            elif isinstance(oe, RateLimitError):
                logger.error("达到速率限制，考虑增加重试次数")
            elif isinstance(oe, APIError):
                logger.error(f"API错误: {oe}")
            raise
        except Exception as e:
            logger.error(f"ask_with_images中的意外错误: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # 不重试TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        使用工具/函数调用向LLM提问并返回响应
        
        参数:
            messages: 对话消息列表
            system_msgs: 可选的系统消息前缀
            timeout: 请求超时时间(秒)
            tools: 要使用的工具列表
            tool_choice: 工具选择策略
            temperature: 响应的采样温度
            **kwargs: 额外的完成参数
            
        返回:
            ChatCompletionMessage: 模型的响应，可能为None
            
        异常:
            TokenLimitExceeded: 如果超出token限制
            ValueError: 如果工具、tool_choice或消息无效
            OpenAIError: API调用失败后重试
            Exception: 意外错误
        """
        try:
            # 验证tool_choice是否有效
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"无效的tool_choice: {tool_choice}")

            # 检查模型是否支持图像
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入token数量
            input_tokens = self.count_message_tokens(messages)

            # 如果有工具，计算工具描述的token数量
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # 检查是否超出token限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 抛出不会被重试的特殊异常
                raise TokenLimitExceeded(error_message)

            # 验证提供的工具
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("每个工具必须是一个包含'type'字段的字典")

            # 设置完成请求参数
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            # 设置模型特定参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            params["stream"] = False  # 工具请求总是使用非流式
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # 检查响应是否有效
            if not response.choices or not response.choices[0].message:
                print(response)
                return None

            # 更新token计数
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # 重新抛出token限制错误而不记录日志
            raise
        except ValueError as ve:
            logger.error(f"ask_tool中的验证错误: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API错误: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("认证失败，请检查API密钥")
            elif isinstance(oe, RateLimitError):
                logger.error("达到速率限制，考虑增加重试次数")
            elif isinstance(oe, APIError):
                logger.error(f"API错误: {oe}")
            raise
        except Exception as e:
            logger.error(f"ask_tool中的意外错误: {e}")
            raise
