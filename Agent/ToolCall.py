import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from Agent.ReAct import ReActAgent
from Infrastructure.exceptions import TokenLimitExceeded
from Infrastructure.logger import logger
from prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from Infrastructure.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from tool import CreateChatCompletion, Terminate, ToolCollection


# 工具调用必须但未提供时的错误消息
TOOL_CALL_REQUIRED = "必须使用工具调用但未提供任何工具"


class ToolCallAgent(ReActAgent):
    """基础工具调用代理类，增强抽象能力，用于处理工具/函数调用"""

    # 代理名称和描述
    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    # 提示词配置
    system_prompt: str = SYSTEM_PROMPT  # 系统提示词
    next_step_prompt: str = NEXT_STEP_PROMPT  # 下一步行动提示词

    # 可用工具集合，默认包含创建聊天完成和终止工具
    # 1. CreateChatCompletion - 创建聊天完成
    # 2. Terminate - 终止工具
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )  
    
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore  
    # 工具选择模式，默认为AUTO(自动选择)，其他可能值：REQUIRED(必须使用工具) / NONE(不使用工具)
    
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])  
    # 特殊工具名称列表，默认只包含终止工具的名称
    # 使用default_factory确保每次实例化时创建新列表，如果直接使用 default=[Terminate().name]，所有实例会共享同一个列表对象

    tool_calls: List[ToolCall] = Field(default_factory=list)  
    # 当前待执行的工具调用列表，初始化为空列表
    
    _current_base64_image: Optional[str] = None  
    # （私有属性）当前处理的base64编码图像，用于存储工具执行产生的图像数据

    max_steps: int = 30  # 最大执行步数限制，防止无限循环
    
    max_observe: Optional[Union[int, bool]] = None  
    # 结果观察长度限制，防止过长的工具输出占用过多内存或干扰LLM处理
    # 可以是：
    # - None: 无限制
    # - int: 自定义最大字符数
    # - bool: False表示禁用观察

    async def think(self) -> bool:
        """处理当前状态并决定下一步行动
        
        返回:
            bool: 是否需要执行行动(True)或只需思考(False)
        """
        # 如果有下一步提示词，添加消息：角色为user，内容为Manus的下一步提示词
        # （注意：实际调用think()在Manus，最后返回的父类ToolCallAgent的think，所以这里的self是Manus）
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # 获取带有工具选项的LLM响应
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )

        except ValueError:
            raise
        except Exception as e:
            # 检查是否是包含TokenLimitExceeded的RetryError
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"达到Token上限，无法继续执行: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        # 解析响应中的工具调用和内容，当LLM调用工具时，content为空，只返回tool_calls
        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # 记录响应信息（只在有内容时显示）
        if content:
            logger.info(f"✨ {self.name}的思考内容: {content}")
        
        # 只在有工具调用时显示工具信息
        if tool_calls:
            logger.info(f"🛠️ {self.name}选择了{len(tool_calls)}个工具")
            logger.info(f"🧰 准备使用的工具: {[call.function.name for call in tool_calls]}")
            logger.info(f"🔧 工具参数: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("未收到LLM的响应")

            # 处理不同的工具选择模式
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(f"🤔 注意，{self.name}尝试在不允许时使用工具!")
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # 创建并添加助手消息
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            # 处理必须使用工具但未提供的情况
            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # 将在act()中处理

            # 自动模式下，如果没有工具调用但有内容则继续
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 糟糕! {self.name}的思考过程遇到问题: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"处理过程中遇到错误: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """执行工具调用并处理结果
        
        返回:
            str: 执行结果的汇总字符串
        """
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # 如果没有工具调用，返回最后一条消息内容
            return self.messages[-1].content or "无内容或可执行命令"

        results = []
        for command in self.tool_calls:
            # 为每个工具调用重置base64图像
            self._current_base64_image = None

            # 执行工具调用
            result = await self.execute_tool(command)

            # 应用观察长度限制
            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 工具'{command.function.name}'执行完成! 结果: {result}"
            )

            # 添加工具响应到记忆
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        # 返回所有结果的合并字符串
        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """执行单个工具调用，包含健壮的错误处理
        
        参数:
            command: 要执行的工具调用对象
            
        返回:
            str: 执行结果或错误信息
        """
        if not command or not command.function or not command.function.name:
            return "错误: 无效的命令格式"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"错误: 未知工具 '{name}'"

        try:
            # 解析参数
            args = json.loads(command.function.arguments or "{}")

            # 添加调试日志
            # logger.debug(f"原始参数: {args}")

            # 确保参数可序列化，处理Union类型
            # 安全处理参数类型
            safe_args = {}
            for k, v in args.items():
                try:
                    if hasattr(v, '__origin__') and isinstance(getattr(v, '__origin__'), type) and v.__origin__ == Union:
                        safe_args[k] = str(v)
                        logger.debug(f"转换Union类型参数: {k}={v}")
                    else:
                        safe_args[k] = v
                except Exception as e:
                    logger.warning(f"参数{k}类型检查异常: {str(e)}")
                    safe_args[k] = str(v)  # 降级处理
            
            args = safe_args

            # 添加转换后日志
            # logger.debug(f"转换后参数: {args}") 
            
            logger.info(f"🔧 正在激活工具: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 处理特殊工具
            await self._handle_special_tool(name=name, result=result)

            # 检查结果是否包含base64图像
            if hasattr(result, "base64_image") and result.base64_image:
                # 存储base64图像供后续使用
                self._current_base64_image = result.base64_image

                # 格式化结果显示
                observation = (
                    f"观察到工具`{name}`的执行结果:\n{str(result)}"
                    if result
                    else f"工具`{name}`执行完成，无输出"
                )
                return observation

            # 标准情况下的结果显示格式化
            observation = (
                f"观察到工具`{name}`的执行结果:\n{str(result)}"
                if result
                else f"工具`{name}`执行完成，无输出"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"解析{name}参数时出错: JSON格式无效"
            logger.error(
                f"📝 参数错误! '{name}'的参数格式无效 - 非法的JSON, 参数:{command.function.arguments}"
            )
            return f"错误: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ 工具'{name}'执行遇到问题: {str(e)}"
            logger.exception(error_msg)
            return f"错误: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具的执行和状态变更
        
        参数:
            name: 工具名称
            result: 执行结果
            kwargs: 其他参数
        """
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # 设置代理状态为已完成
            logger.info(f"🏁 特殊工具'{name}'已完成任务!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """判断工具执行是否应该结束代理
        
        返回:
            bool: 是否应该结束执行
        """
        return True

    def _is_special_tool(self, name: str) -> bool:
        """检查工具名称是否在特殊工具列表中
        
        参数:
            name: 要检查的工具名称
            
        返回:
            bool: 是否是特殊工具
        """
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """清理代理工具使用的资源"""
        logger.info(f"🧹 正在清理代理'{self.name}'的资源...")
        self.memory.clear()
        logger.info(f"🧹 代理'{self.name}'的记忆已清空.")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 正在清理工具: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 清理工具'{tool_name}'时出错: {e}", exc_info=True
                    )
        logger.info(f"✨ 代理'{self.name}'清理完成.")

    async def run(self, request: Optional[str] = None) -> str:
        """运行代理并在完成后清理资源
        
        参数:
            request: 可选初始请求字符串
            
        返回:
            str: 运行结果
        """
        try:
            logger.debug(f"🔄 {self.name}开始执行")
            result = await super().run(request)
            await self.cleanup()
            return result
        except Exception as e:
            logger.error(f"代理执行异常: {str(e)}")
            raise
        finally:
            logger.debug(f"✨ {self.name}执行结束")