# flow/planning.py

import json
import time
from typing import Dict, List, Optional, Union, Any

from pydantic import Field

from Agent.Base import BaseAgent
from flow.base import BaseFlow
from llm import LLM
from tool import PlanningTool
from tool.planning import Status as PlanStepStatus
from Infrastructure.logger import logger
from Infrastructure.schema import AgentState, Message, ToolChoice


class PlanningFlow(BaseFlow):
    """规划流程类，管理任务规划与多代理执行"""

    llm: LLM = Field(default_factory=lambda: LLM())  # 语言模型实例
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)  # 规划工具实例
    executor_keys: List[str] = Field(default_factory=list)  # 执行代理键名列表
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")  # 当前活动计划ID
    current_step_index: Optional[int] = None  # 当前执行步骤索引

    def __init__(
        self, 
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], 
        **data
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
                # 获取计划信息
                plan_result = await self.planning_tool.execute(
                    command="get",
                    plan_id=self.active_plan_id
                )

                # 验证计划是否创建成功
                if not plan_result.output:
                    logger.error(f"计划创建失败。计划ID {self.active_plan_id} 未找到")
                    return f"创建计划失败: {input_text}"

            result = ""
            while True:
                # 获取当前步骤索引（int）和步骤信息（一个字典）
                step_index, step_info = await self._get_current_step_info()

                # 如果没有更多步骤或计划已完成，退出循环，总结计划结果
                if step_index is None or not step_info:
                    result += await self._finalize_plan()
                    break

                # 更新当前步骤索引
                self.current_step_index = step_index  

                # 使用合适的代理执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)

                # 执行步骤并获取结果
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 总结当前步骤的实际执行结果，并更新计划文本
                plan_result = await self._update_plan_text(step_result)
                # 显示更新后的计划
                logger.info(f"\n📋 更新后的计划状态:\n{plan_result}")
                
                # 检查代理是否想终止流程
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            # 完成计划后将结果存储在execution_log中
            plan_data = self.planning_tool.plans[self.active_plan_id] # plan_data是一个字典
            plan_data.execution_log = result # 将执行结果存储在execution_log中
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
            "作为专业规划助手，请按以下规则创建可执行的简明计划：\n"
            "1. 分析任务需求(分析显性需求和隐性需求)，明确任务的核心目标及成功标准"
            "2. 计划应包含明确的阶段，如 学习准备→开发→测试与优化→文档记录\n"
            "3. 必须使用[PHASE]标记各个步骤所处的阶段，如[RESEARCH]/[DEV]/[OPTIMIZATION]/[DOCUMENTATION]\n"
            "4. 计划应包含清晰的可执行步骤，每个步骤应包括描述和预期输出，但'Focus on key milestones rather than detailed sub-steps.\n"
            "示例步骤格式：\n"
            "[RESEARCH] 理解相关基础知识与算法原理"
        )

        # 创建用户消息
        user_message = Message.user_message(
            f"请在分析用户显性需求和隐性需求后，调用Plan工具为以下任务创建执行计划：\n"
            f"注意事项：各个步骤的开头必须使用[PHASE]标记，如[RESEARCH]/[DEV]/[OPTIMIZATION]/[DOCUMENTATION]\n"
            f"任务需求：{request}\n\n"
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

        # 调用planningtool创建默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": [

                    {
                        "description": "[RESEARCH] 信息收集",
                        "expected_output": "获取执行任务所需的全部数据"
                    },
                    {
                        "description": "[EXECUTE] 任务执行",
                        "expected_output": "完成主要任务交付物"
                    },
                    {
                        "description": "[OPTIMIZE] 优化改进",
                        "expected_output": "根据反馈进行必要的优化"
                    },
                    {
                        "description": "[VALIDATE] 结果验证",
                        "expected_output": "确认结果符合质量要求"
                    },
                    {
                        "description": "[DOCUMENT] 文档记录",
                        "expected_output": "记录执行过程与结果"
                    }
                ],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """获取当前步骤信息
        
        返回:
            元组(步骤索引, 步骤信息)，如果没有活动步骤则返回(None, None)
        """
        # 检查计划是否存在
        if not self.active_plan_id or self.active_plan_id not in self.planning_tool.plans:
            logger.error(f"计划ID {self.active_plan_id} 未找到")
            return None, None

        try:
            # 获取计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id] # plan_data是一个字典
            steps = plan_data.steps # steps是一个列表

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):  # 遍历所有步骤，i是索引，step是步骤内容
                # 从StepInfo对象直接获取状态
                if step.status in PlanStepStatus.get_active_statuses():
                    # 提取步骤类型(如[SEARCH]或[CODE])
                    import re
                    type_match = re.search(r"$$([A-Z_]+)$$", step.description)
                    step_type = type_match.group(1).lower() if type_match else None
                    # 步骤描述格式化
                    step_info = {
                        "text": step.description,
                        "expected_output": step.expected_output,
                        "current_status": step.status,
                        "notes": step.notes, # 该步骤的备注，
                        "type": step_type, # 新增：步骤类型，对应适配的Agent
                        "actual_result":step.actual_result # 实际输出
                    }

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
                        step.status = PlanStepStatus.IN_PROGRESS.value

                    return i, step_info # 返回当前步骤索引（int）和步骤信息（一个字典）

            return None, None  # 未找到可执行的步骤

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
        # 确保浏览器代理已初始化
        if hasattr(executor, 'browser_context_helper'):
            try:
                # 在步骤执行前初始化浏览器上下文
                await executor.browser_context_helper.ensure_initialized()
            except Exception as e:
                logger.error(f"浏览器初始化失败: {str(e)}")
                return f"浏览器初始化失败: {str(e)}"

        # 准备计划状态上下文
        plan_context = await self._get_plan_text()
        step_text = step_info.get("text", f"步骤 {self.current_step_index}")
        expected_output = step_info.get("expected_output", "未定义")

        # 创建步骤执行提示
        step_prompt = f"""
        << 执行约束 >>
        1. 专注当前步骤：你只能处理步骤{self.current_step_index}，禁止操作后续步骤
        2. 超时控制：若3分钟内无实质性进展，自动标记为阻塞
        3. 结果验证：必须严格对比实际结果与下列预期输出
        4. 依赖检查：确认前置步骤{self.current_step_index-1}已100%完成

        << 计划上下文 >>
        {plan_context}

        << 当前任务 >>
        ■ 步骤编号：{self.current_step_index+1}/{len(self.planning_tool.plans[self.active_plan_id].steps)}
        ■ 任务描述：{step_text}
        ■ 预期输出：{expected_output}
        ■ 备注：{step_info.get('notes', '无备注')}

        << 执行策略 >>
        1. 分阶段执行：
        - Phase 1：执行核心操作（使用必要工具）
        - Phase 2：生成结构化结果（JSON格式）
        - Phase 3：差异分析（实际vs预期）
        
        2. 质量控制：
        ! 当实际结果匹配度不高（<20%）时：
            a) 自动重试(最多3次) 
            b) 仍失败则标记为阻塞
        
        3. 过程监控：
        √ 检测重复/循环执行模式

        << 结果评估标准 >>
        评估维度       | 合格标准
        -------------------------------
        完整性        | 覆盖所有需求要点
        准确性        | 关键数据误差率<10%
        一致性        | 结果与预期基本一致
        可交付性      | 可直接用于下一步骤
        合规性        | 符合预定义格式要求

        << 输出要求 >>
        输出markdown格式的执行结果。
        """

        # 使用代理执行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 标记步骤为已完成
            await self._mark_step_completed()

            # 在清理前确保浏览器上下文仍然有效
            if hasattr(executor, 'browser_context_helper'):
                try:
                    await executor.browser_context_helper.ensure_initialized()
                except Exception as e:
                    logger.warning(f"浏览器上下文维护失败: {str(e)}")

            return step_result
        except Exception as e:
            logger.error(f"执行步骤 {self.current_step_index+1}/{len(self.planning_tool.plans[self.active_plan_id].steps)} 时出错: {e}")
            return f"执行步骤 {self.current_step_index+1}/{len(self.planning_tool.plans[self.active_plan_id].steps)} 时出错: {str(e)}"

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
            # 添加已完成步骤的执行日志
            plan = self.planning_tool.plans[self.active_plan_id]
            step = plan.steps[self.current_step_index]
            step.notes += f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  执行状态: {PlanStepStatus.COMPLETED.value}"
            # 显示进度
            logger.info(
                f"已标记步骤 {self.current_step_index+1}/{len(self.planning_tool.plans[self.active_plan_id].steps)} 为COMPLETED"
                )
            
        except Exception as e:
            logger.warning(f"更新计划状态失败: {e}")
            # 直接更新规划工具存储中的状态
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.step_statuses

                # 确保步骤状态列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data.step_statuses = step_statuses

    async def _update_plan_text(self, step_result: str) -> str:
        """总结当前步骤的实际执行结果，更新并返回计划文本"""
        try:
            plan_data = self.planning_tool.plans[self.active_plan_id] # plan_data是一个字典
            steps = plan_data.steps # 字典的 get() 方法安全获取键的对应值，steps是一个列表，元素为StepInfo对象

            # 利用llm总结步骤结果，提取精准简要的有效信息
            system_message = Message.system_message(
                "【结果分析】"
                "任务：按以下思考顺序从原始执行日志中总结关键信息：\n"
                "1. 分析总结步骤操作内容（信息收集/数据处理/系统操作/代码编写……）\n"
                "2. 提取关键元信息（URL/操作对象/关键参数/执行结果……）\n"
                "3. 标记为[SUCCESS/ERROR/WARNING]并总结错误信息（保留原始错误码+核心描述（20字内））\n"
                "4. 记录状态变更（起始值 → 结束值（带时间戳则保留））"
            )
            summary_prompt = Message.user_message(
                f"""
                原始执行日志：
                {step_result}

                请按上述思考顺序总结整个流程的关键信息，确保输出简洁且包含关键元信息。
                输出格式：
                整个流程的操作内容：[SUCCESS/ERROR/WARNING] 操作内容（50字内）
                关键元信息：
                - URL/操作对象/关键参数/执行结果……
                错误信息（无错误则不显示）：
                - 错误码：描述（20字内）

                注意：
                1. 关键元信息处理规则：
                    - URL：保留完整路径并去重
                    - API调用：显示端点+关键参数
                    - 工具操作：显示工具名称+关键参数+操作内容+影响对象
                    - 代码编写：显示文件路径+代码作用总结+影响对象
                2. 确保输出简洁明了，重点突出关键信息
                3. 避免重复信息，仅保留必要的元信息
                4. 确保输出格式清晰，便于后续处理
                """
            )
            summary_result = await self.llm.ask(
                messages=[summary_prompt], 
                system_msgs=[system_message]
            )

            # 更新步骤的实际结果
            steps[self.current_step_index].actual_result = summary_result
            # 提取更新后的完整计划
            plan_result = await self.planning_tool.execute(
                command="get",
                plan_id=self.active_plan_id
            )
            # 返回更新后的计划文本
            return plan_result.output
        except Exception as e:
            logger.error(f"更新计划文本时出错: {e}")
            return f"更新计划文本时出错: {str(e)}"
        
    async def _get_plan_text(self) -> str:
        """获取当前计划的格式化文本
        
        返回:
            计划状态文本，（PlanningTool中的类方法_format_plan的输出格式）
        异常：
            从存储中生成计划文本
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
        """从存储中生成计划文本(PlanningTool的get command失败时使用)
        
        返回:
            格式化后的计划文本
        """
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"错误: 未找到计划ID {self.active_plan_id}"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            output = []

            # 头部信息
            output.append(f"📋 计划: {plan_data.title} (ID: {self.active_plan_id})")
            output.append("-" * 50)

            # 进度统计
            steps = plan_data.steps
            total = len(steps)
            status_counts = {
                "completed": 0,
                "in_progress": 0,
                "blocked": 0,
                "not_started": 0
            }
            
            current_step = None
            blocked_steps = []
            
            # 状态检测循环
            for idx, step in enumerate(steps):
                status = step.status.lower()  # 确保状态小写
                
                # 状态计数
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    logger.warning(f"无效的状态值: {status} 于步骤 {idx+1}")
                    continue

                # 检测阻塞步骤
                if status == PlanStepStatus.BLOCKED.value:
                    blocked_steps.append(idx)

                # 确定当前步骤
                if current_step is None:
                    if status == PlanStepStatus.IN_PROGRESS.value:
                        current_step = idx
                    elif status == PlanStepStatus.NOT_STARTED.value:
                        current_step = idx

            # 进度显示
            output.append(f"进度: {status_counts['completed']}/{total} 步骤完成")
            output.append(f"├── 完成( ✅ ): {status_counts['completed']}")
            output.append(f"├── 进行中( 🚧 ): {status_counts['in_progress']}")
            output.append(f"├── 阻塞( ⚠️ ): {status_counts['blocked']}")
            output.append(f"└── 未开始( ⏳ ): {status_counts['not_started']}\n")

            # 详细步骤列表
            output.append("📝 步骤详情:")
            for idx, step in enumerate(steps):
                status_icon = self._status_emoji(step.status)
                prefix = "➤" if idx == current_step else "•"
                
                # 基础信息
                output.append(f"{prefix} [{status_icon}] 步骤 {idx+1}: {step.description}")
                
                # 状态详细信息
                if step.status != PlanStepStatus.NOT_STARTED.value:
                    info_lines = []
                    info_lines.append(f"    ├── 状态: {step.status}")
                    if step.expected_output:
                        info_lines.append(f"    ├── 预期: {step.expected_output}")
                    if step.notes:
                        info_lines.append(f"    ├── 备注: {step.notes}")
                    if step.actual_result is not None:
                        info_lines.append(f"    └── 实际: \n{self._format_result(step.actual_result)}")
                    
                    # 优化显示结构
                    if len(info_lines) > 1:
                        info_lines[-1] = info_lines[-1].replace("├──", "└──")
                    output.extend(info_lines)
                output.append("")  # 步骤间空行

            # 当前步骤强调
            if current_step is not None and current_step < len(steps):
                step = steps[current_step]
                output.append("🔍 当前应执行步骤:")
                output.append(f"   → 步骤 {current_step+1}: {step.description}")
                output.append(f"      预期输出: {step.expected_output or '未指定'}")
                if step.actual_result is not None:
                    output.append(f"      实际结果: {self._format_result(step.actual_result)}")
                output.append(f"      状态: {self._status_emoji(step.status)} {step.status}")  
                if step.notes:
                    output.append(f"      备注: {step.notes}")
                output.append("")  # 空行分隔

            # 阻塞步骤警告
            if blocked_steps:
                output.append("🚨 阻塞步骤需要立即处理:")
                for idx in blocked_steps:
                    step = steps[idx]
                    output.append(f"   ⚠ 步骤 {idx+1}: {step.description}")
                    output.append(f"      阻塞原因: {step.notes or '未说明原因'}")
                output.append("")  # 空行分隔

            # 执行约束说明
            ###########还有修改的空间##########
            output.append("\n⚠️  执行注意事项:")
            output.append("1. 严格按步骤顺序执行，当前步骤未完成前禁止处理后续步骤，你只需要结合之前的步骤信息，执行当前应执行步骤")
            output.append("2. 遇到阻塞状态( ⚠️ )必须优先解决，解除阻塞前不得继续后续步骤")
            output.append("3. 实际结果与预期不符时需重新执行当前步骤")
            output.append("-" * 50)

            return "\n".join(output)

        except Exception as e:
            logger.error(f"从存储生成计划文本时出错: {e}")
            return f"错误: 无法检索计划ID {self.active_plan_id}"

    # _generate_plan_text_from_storage的辅助方法
    def _status_emoji(self, status: str) -> str:
        """获取状态对应的表情符号"""
        return {
            "completed": "✅",
            "in_progress": "🚧",
            "blocked": "⚠️",
            "not_started": "⏳"
        }.get(status.lower(), "❓")
    def _format_result(self, result: Any) -> str:
        """格式化实际结果"""
        if isinstance(result, Exception):
            return f"错误: {str(result)}"
        if result is None:
            return "暂无记录"
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

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
                f"""
                以下是最终计划状态:\n\n{plan_text}\n\n
                请按顺序提供:
                1. 已完成工作的完成状态摘要
                2. 各个步骤的执行情况以及与其预期结果的对比分析
                3. 分析计划整体执行情况以及不足之处，并给出大概的可执行的简单的优化方向
                4. 最终说明
                """
            )

            response = await self.llm.ask(
                messages=[user_message], 
                system_msgs=[system_message]
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