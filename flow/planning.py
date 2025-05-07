import json
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from Agent.Base import BaseAgent
from flow.base import BaseFlow
from llm import LLM
from tool import PlanningTool
from Infrastructure.logger import logger
from Infrastructure.schema import AgentState, Message, ToolChoice


class PlanStepStatus(str, Enum):
    """计划步骤状态枚举类"""
    NOT_STARTED = "not_started"  # 未开始
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"  # 已完成
    BLOCKED = "blocked"  # 已阻塞

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """获取所有可能的状态值列表"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """获取活动状态列表(未开始或进行中)"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """获取状态与标记符号的映射关系"""
        return {
            cls.COMPLETED.value: "[✓]",  # 完成标记
            cls.IN_PROGRESS.value: "[→]",  # 进行中标记
            cls.BLOCKED.value: "[!]",  # 阻塞标记
            cls.NOT_STARTED.value: "[ ]",  # 未开始标记
        }


class PlanningFlow(BaseFlow):
    """规划流程类，管理任务规划与多代理执行"""

    llm: LLM = Field(default_factory=lambda: LLM())  # 语言模型实例
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)  # 规划工具实例
    executor_keys: List[str] = Field(default_factory=list)  # 执行代理键名列表
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")  # 当前活动计划ID
    current_step_index: Optional[int] = None  # 当前执行步骤索引

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        """初始化规划流程
        
        参数:
            agents: 可接受单个代理、代理列表或代理字典
            **data: 其他初始化数据
        """
        # 处理执行代理键名
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 处理计划ID
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 初始化规划工具
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 调用父类初始化
        super().__init__(agents, **data)

        # 默认使用所有代理作为执行代理
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """获取适合当前步骤的执行代理
        
        参数:
            step_type: 步骤类型，用于选择特定代理
            
        返回:
            选中的代理实例
        """
        # 如果步骤类型匹配代理键名，使用该代理
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 否则使用第一个可用执行代理或回退到主代理
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 最终回退到主代理
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """执行规划流程
        
        参数:
            input_text: 输入文本/请求
            
        返回:
            执行结果文本
        """
        try:
            if not self.primary_agent:
                raise ValueError("无可用主代理")

            # 根据输入创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 验证计划是否创建成功
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(f"计划创建失败。计划ID {self.active_plan_id} 未找到")
                    return f"创建计划失败: {input_text}"

            result = ""
            while True:
                # 获取当前执行步骤信息
                self.current_step_index, step_info = await self._get_current_step_info()

                # 如果没有更多步骤或计划已完成，退出循环
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 使用合适的代理执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 检查代理是否想终止流程
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"规划流程执行错误: {str(e)}")
            return f"执行失败: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """创建初始计划
        
        参数:
            request: 用户请求文本
        """
        logger.info(f"正在创建初始计划，ID: {self.active_plan_id}")

        # 创建系统消息
        system_message = Message.system_message(
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )

        # 创建用户消息
        user_message = Message.user_message(
            f"创建一个合理的计划来完成以下任务: {request}"
        )

        # 调用LLM创建计划
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 处理工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析参数
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"工具参数解析失败: {args}")
                            continue

                    # 设置计划ID并执行工具
                    args["plan_id"] = self.active_plan_id
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"计划创建结果: {str(result)}")
                    return

        # 如果执行到这里，创建默认计划
        logger.warning("计划创建失败，正在创建默认计划")

        # 使用工具集合创建默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """获取当前步骤信息
        
        返回:
            元组(步骤索引, 步骤信息)，如果没有活动步骤则返回(None, None)
        """
        if not self.active_plan_id or self.active_plan_id not in self.planning_tool.plans:
            logger.error(f"计划ID {self.active_plan_id} 未找到")
            return None, None

        try:
            # 从规划工具存储中获取计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤信息
                    step_info = {"text": step}

                    # 尝试从文本中提取步骤类型(如[SEARCH]或[CODE])
                    import re
                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 标记当前步骤为进行中
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"标记步骤为进行中时出错: {e}")
                        # 直接更新步骤状态
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # 未找到活动步骤

        except Exception as e:
            logger.warning(f"查找当前步骤索引时出错: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """执行当前步骤
        
        参数:
            executor: 执行代理
            step_info: 步骤信息
            
        返回:
            步骤执行结果文本
        """
        # 准备计划状态上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"步骤 {self.current_step_index}")

        # 创建步骤执行提示
        step_prompt = f"""
        当前计划状态:
        {plan_status}

        你的当前任务:
        你正在处理步骤 {self.current_step_index}: "{step_text}"

        请使用适当的工具执行此步骤。完成后，请提供你所完成工作的摘要。
        """

        # 使用代理执行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 标记步骤为已完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"执行步骤 {self.current_step_index} 时出错: {e}")
            return f"执行步骤 {self.current_step_index} 时出错: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """标记当前步骤为已完成"""
        if self.current_step_index is None:
            return

        try:
            # 通过规划工具标记步骤
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(f"已标记步骤 {self.current_step_index} 为已完成")
        except Exception as e:
            logger.warning(f"更新计划状态失败: {e}")
            # 直接更新规划工具存储中的状态
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保步骤状态列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """获取当前计划的格式化文本
        
        返回:
            计划状态文本
        """
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"获取计划时出错: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """从存储中生成计划文本(规划工具失败时使用)
        
        返回:
            格式化后的计划文本
        """
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"错误: 未找到计划ID {self.active_plan_id}"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "未命名计划")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保步骤状态和注释与步骤数量匹配
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 统计各状态步骤数量
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}
            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            # 构建计划文本
            plan_text = f"计划: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"
            plan_text += f"进度: {completed}/{total} 步骤完成 ({progress:.1f}%)\n"
            plan_text += f"状态: {status_counts[PlanStepStatus.COMPLETED.value]} 完成, "
            plan_text += f"{status_counts[PlanStepStatus.IN_PROGRESS.value]} 进行中, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} 阻塞, "
            plan_text += f"{status_counts[PlanStepStatus.NOT_STARTED.value]} 未开始\n\n"
            plan_text += "步骤:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(zip(steps, step_statuses, step_notes)):
                status_mark = status_marks.get(status, status_marks[PlanStepStatus.NOT_STARTED.value])
                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   备注: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"从存储生成计划文本时出错: {e}")
            return f"错误: 无法检索计划ID {self.active_plan_id}"

    async def _finalize_plan(self) -> str:
        """完成计划并生成摘要
        
        返回:
            计划完成摘要文本
        """
        plan_text = await self._get_plan_text()

        # 使用LLM生成摘要
        try:
            system_message = Message.system_message(
                "你是一个规划助手。你的任务是总结已完成计划。"
            )

            user_message = Message.user_message(
                f"计划已完成。以下是最终计划状态:\n\n{plan_text}\n\n请提供已完成工作的摘要和最终想法。"
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"计划完成:\n\n{response}"
        except Exception as e:
            logger.error(f"使用LLM完成计划时出错: {e}")

            # 回退使用代理生成摘要
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                计划已完成。以下是最终计划状态:

                {plan_text}

                请提供已完成工作的摘要和最终想法。
                """
                summary = await agent.run(summary_prompt)
                return f"计划完成:\n\n{summary}"
            except Exception as e2:
                logger.error(f"使用代理完成计划时出错: {e2}")
                return "计划完成。生成摘要时出错。"