# tool/planning.py
import json
from typing import Dict, List, Literal, Optional, Any

from pydantic import ValidationError, Field

from Memory.GlobalMemory import GlobalMemory
from Infrastructure.exceptions import ToolError
from tool.base import BaseTool, ToolResult
from Infrastructure.schema import Status, StepInfo, Plan

_PLANNING_TOOL_DESCRIPTION = """
A planning tool enabling agents to create and manage multi-step plans for complex problem solving. Key features include:
- Create new plans (with titles, detailed steps, expected outputs)
- Update existing plans (intelligent status merging)
- View plan lists/details (formatted output)
- Set active plan
- Mark step statuses (in_progress/completed/blocked)
- Track overall progress and step-level execution details
- Auto-detect blocked steps and execution order constraints
Provides structured data storage, visual progress tracking, and JSON-formatted results.
"""
"""
一个规划工具，允许代理创建和管理用于解决复杂任务的多步骤计划。该工具支持以下功能：
- 创建新计划（包含标题、详细步骤及预期输出）
- 更新现有计划内容（智能合并步骤状态）
- 查看计划列表及详细信息（支持格式化输出）
- 设置当前活动计划
- 标记步骤状态（进行中/已完成/阻塞）
- 跟踪计划整体进度和步骤级执行详情
- 自动检测阻塞步骤和执行顺序约束
提供结构化数据存储和可视化进度跟踪，支持JSON格式结果输出。
"""

class PlanningTool(BaseTool):
    """
    规划工具类，用于创建多步骤计划 和 管理多步骤执行计划
    """
    
    # 工具元数据定义
    name: str = "planning"  # 工具名称，用于系统识别
    description: str = _PLANNING_TOOL_DESCRIPTION  # 工具功能描述
    global_memory: GlobalMemory = Field(...)  # 必需字段

    # 计划存储结构定义
    plans: Dict[str, Plan] = Field(
        default_factory=dict, 
        description="所有计划的存储仓库"
    )  # 所有计划的存储仓库，格式为嵌套着Plan对象的字典: {plan_id: {plan_data}}
    _current_plan_id: Optional[str] = None  # 当前活动计划的ID

    def __init__(
        self,
        global_memory: Optional[GlobalMemory] = None,
        **kwargs
    ):
        # 确保global_memory存在
        if global_memory is None:
            # 获取单例实例
            global_memory = GlobalMemory()

        # 直接调用父类初始化并传递参数
        super().__init__(
            global_memory=global_memory,  # 确保传递必需字段
            **kwargs
        )
        
        # 显式设置 global_memory
        self.global_memory = global_memory
    
    # OpenAPI规范的参数定义 (用于LLM工具调用)
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "Operation command to execute. Available: create, update, list, get, set_active, mark_step, delete",
                "enum": ["create", "update", "list", "get", "set_active", "mark_step", "delete"],
                "type": "string"
            },
            "plan_id": {
                "description": "Unique plan identifier. Required for create/update/set_active/delete, optional for get/mark_step (uses active plan)",
                "type": "string"
            },
            "title": {
                "description": "Plan title (required for create, optional for update)",
                "type": "string"
            },
            "steps": {
                "description": "List of plan steps with descriptions and expected outputs (required for create, optional for update)",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "Step description"},
                        "expected_output": {"type": "string", "description": "Expected completion outcome"},
                        "notes": {"type": "string", "description": "Additional notes"}
                    },
                    "required": ["description", "expected_output"],
                    "additionalProperties": False
                }
            },
            "step_index": {
                "description": "Step index to operate on (0-based, required for mark_step)",
                "type": "integer"
            },
            "step_status": {
                "description": "Target status for step (used with mark_step)",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string"
            },
            "step_notes": {
                "description": "Additional notes for step (optional with mark_step)",
                "type": "string"
            }
        },
        "required": ["command"],
        "additionalProperties": False
    }

    async def execute(
        self,
        *,
        command: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[dict]] = None, # llm只能返回字典列表，所以这里也只能接收字典列表
        step_index: Optional[int] = None,
        step_status: Optional[Literal[
            Status.NOT_STARTED.value, 
            Status.IN_PROGRESS.value, 
            Status.COMPLETED.value, 
            Status.BLOCKED.value
        ]] = None,
        step_notes: Optional[str] = None,
        **kwargs,
    ):
        """
        PlanningTool的执行入口，根据命令执行不同的操作
        
        参数:
            command: 要执行的操作
            plan_id: 计划的唯一标识符
            title: 计划标题
            steps: 计划步骤字典列表  # llm只能返回字典列表，所以这里也只能接收字典列表
            step_index: 步骤索引
            step_status: 步骤状态（使用Status常量）
            step_notes: 步骤备注
        """
        if command == "create":
            return self._create_plan(plan_id, title, steps)
        elif command == "update":
            return self._update_plan(plan_id, title, steps)
        elif command == "list":
            return self._list_plans()
        elif command == "get":
            return self._get_plan(plan_id)
        elif command == "set_active":
            return self._set_active_plan(plan_id)
        elif command == "mark_step":
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        elif command == "delete":
            return self._delete_plan(plan_id)
        else:
            raise ToolError(
                f"无法识别的命令: {command}。允许的命令有: create, update, list, get, set_active, mark_step, delete"
            )

    def _create_plan(
            self, 
            plan_id: str, 
            title: str, 
            steps: List[dict], # llm只能返回字典列表，所以这里也只能接收字典列表
    ) -> ToolResult:
        """
        创建新计划
        逻辑流程:
        1. 参数校验 → 2. 构建步骤对象 → 3. 存储计划，同步到全局记忆 → 4. 设为活动计划 → 5. 返回结果
        """
        # 参数校验
        if not plan_id:
            raise ToolError("create命令需要plan_id参数")
        if plan_id in self.plans:
            raise ToolError(
                f"计划ID '{plan_id}'已存在。使用'update'命令修改现有计划"
            )
        if not title:
            raise ToolError("create命令需要title参数")
        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, dict) for step in steps)  # llm只能返回字典列表，所以这里也只能接收字典列表
        ):
            raise ToolError(
                "create命令需要steps参数，且必须是非空字典列表"  # llm只能返回字典列表，所以这里也只能接收字典列表
            )
        
        # 构建步骤对象
        step_objects = []
        for i, step in enumerate(steps):  # llm只能返回字典列表，所以这里也只能接收字典列表
            try:
                step_obj = StepInfo(**step)
                step_objects.append(step_obj)
            except ValidationError:
                raise ToolError(f"无效的步骤格式: 索引{i} - {step}")    

        # 创建计划结构
        plan = Plan(
            plan_id=plan_id,         # 计划唯一标识
            title=title,             # 标题
            steps=step_objects,      # 列表（各个元素为StepInfo对象）
            execution_log=""         # 初始化执行日志
        )

        self.plans[plan_id] = plan
        self.global_memory.sync_plans(self.plans)  # 同步到全局记忆
        self._current_plan_id = plan_id  # 设为活动计划

        return ToolResult(
            output=f"计划创建成功，ID: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
            self, 
            plan_id: str, 
            title: Optional[str], 
            steps: Optional[List[dict]]
    ) -> ToolResult:
        """
        更新计划
        逻辑流程:
        1. 校验planID → 2. 更新标题 → 3. 更新步骤 → 4. 返回结果
        实现要点:
        - 标题更新: 直接替换
        - 步骤更新: 智能合并状态和备注
           - 位置相同的未修改步骤保留原状态
           - 新增/修改的步骤重置为未开始状态
        """
        # 参数校验
        if not plan_id:
            raise ToolError("update命令需要plan_id参数")
        if plan_id not in self.plans:
            raise ToolError(f"找不到ID为 {plan_id} 的计划")
        plan = self.plans[plan_id]

        # 更新标题
        if title:
            plan.title = title

        # 更新步骤内容
        if steps:
            # 参数校验
            if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
                raise ToolError("update命令的steps参数必须是非空字典列表")

            # 创建最终步骤列表
            final_steps = []
            old_steps = plan.steps
            
            # 处理每个新步骤
            for i, step_dict in enumerate(steps):
                try:
                    # 创建新步骤对象（使用传入的字典）
                    new_step = StepInfo(
                        description=step_dict.get("description", ""),
                        expected_output=step_dict.get("expected_output", ""),
                        notes=step_dict.get("notes", "")  # 使用传入的备注
                    )
                    
                    # 检查是否存在对应旧步骤
                    if i < len(old_steps):
                        old_step = old_steps[i]
                        
                        # 比较步骤内容（只比较描述和预期输出）
                        if (new_step.description == old_step.description and 
                            new_step.expected_output == old_step.expected_output):
                            # 内容相同：继承状态和实际结果
                            new_step.status = old_step.status
                            new_step.actual_result = old_step.actual_result
                        else:
                            # 内容不同：重置状态和实际结果
                            new_step.status = Status.NOT_STARTED.value
                            new_step.actual_result = None
                    else:
                        # 新增步骤：使用默认状态
                        new_step.status = Status.NOT_STARTED.value
                        new_step.actual_result = None
                    
                    final_steps.append(new_step)
                except ValidationError as e:
                    raise ToolError(f"步骤{i}配置错误: {str(e)}")
        
            # 更新计划数据
            plan.steps = final_steps

        return ToolResult(
            output=f"计划更新成功: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self) -> ToolResult:
        """列出所有可用计划"""
        if not self.plans:
            return ToolResult(
                output="没有可用计划。使用'create'命令创建新计划"
            )

        output = "可用计划:\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (当前活动)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for step in plan.steps 
                if step.status == Status.COMPLETED.value  # 使用状态常量
            )
            # 类型检查,确保数据结构正确
            if not all(isinstance(step, StepInfo) for step in plan.steps):
                raise ToolError(f"计划 {plan_id} 包含无效的步骤数据类型")
            total = len(plan.steps)
            progress = f"{completed}/{total} 步骤完成"
            output += f"• {plan_id}{current_marker}: {plan.title} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        """获取特定计划的详细信息，
        
        返回：
            ToolResult: 包含计划详细信息的结果对象（类方法_format_plan的输出格式）
        """
        if not plan_id:
            # 未指定plan_id时使用当前活动计划
            if not self._current_plan_id:
                raise ToolError(
                    "没有活动计划。请指定plan_id或设置活动计划"
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"找不到ID为 {plan_id} 的计划")

        plan = self.plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    def _set_active_plan(self, plan_id: Optional[str]) -> ToolResult:
        """设置活动计划"""
        if not plan_id:
            raise ToolError("set_active命令需要plan_id参数")

        if plan_id not in self.plans:
            raise ToolError(f"找不到ID为 {plan_id} 的计划")

        self._current_plan_id = plan_id
        return ToolResult(
            output=f"计划 '{plan_id}' 已设为活动计划\n\n{self._format_plan(self.plans.plan_id)}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        """
        标记步骤状态
        逻辑流程:
        1. 校验参数 → 2. 查找计划 → 3. 标记步骤 → 4. 返回结果
        """
        # 输入参数检验（plan）
        if not plan_id:
            # 未指定plan_id时使用当前活动计划
            if not self._current_plan_id:
                raise ToolError(
                    "没有活动计划。请指定plan_id或设置活动计划"
                )
            plan_id = self._current_plan_id
        if plan_id not in self.plans:
            raise ToolError(f"找不到ID为 {plan_id} 的计划")
        if step_index is None:
            raise ToolError("mark_step命令需要step_index参数")

        plan = self.plans[plan_id]

        # 输入参数检验（step_index）
        if step_index < 0 or step_index >= len(plan.steps):
            raise ToolError(
                f"无效的step_index: {step_index}。有效范围: 0 到 {len(plan.steps)-1}"
            )
        # 输入参数检验（step_status、step_notes）
        if step_status and step_status not in [
            Status.NOT_STARTED.value,
            Status.IN_PROGRESS.value,
            Status.COMPLETED.value,
            Status.BLOCKED.value
        ]:
            raise ToolError(
                f"无效的step_status: {step_status}。有效状态: not_started, in_progress, completed, blocked"
            )

        step = plan.steps[step_index]

        # 更新步骤状态和备注
        if step_status:
            step.status = step_status
        if step_notes:
            step.notes = step_notes

        # 返回结果
        return ToolResult(
            output=f"步骤状态已更新\n{self._format_plan(plan)}"
        )

    def _delete_plan(self, plan_id: Optional[str]) -> ToolResult:
        """删除计划"""
        if not plan_id:
            raise ToolError("delete命令需要plan_id参数")

        if plan_id not in self.plans:
            raise ToolError(f"找不到ID为 {plan_id} 的计划")

        del self.plans[plan_id]

        # 如果删除的是活动计划，清除活动计划标记
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return ToolResult(output=f"计划 '{plan_id}' 已删除")

    def _format_plan(self, plan: Dict) -> str:
        """格式化计划信息，强调执行顺序和当前步骤状态"""
        output = []

        # 头部信息
        output.append(f"📋 计划: {plan.title} (ID: {plan.plan_id})")
        output.append("-" * 50)

        # 进度统计
        steps = plan.steps
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
            status = step.status.lower()
            
            # 状态计数
            if status in status_counts:
                status_counts[status] += 1
            
            # 检测阻塞步骤
            if status == Status.BLOCKED.value:
                blocked_steps.append(idx)
            
            # 确定当前步骤（仅第一次出现）
            if current_step is None:
                if status == Status.IN_PROGRESS.value:
                    current_step = idx
                elif status == Status.NOT_STARTED.value:
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
            
            # 状态详细信息（仅显示非未开始状态）
            if step.status != Status.NOT_STARTED.value:
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

        # 当前步骤强调（在所有步骤之后显示）
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

    def _status_emoji(self, status: str) -> str:
        """状态符号可视化"""
        return {
            "completed": "✅",
            "in_progress": "🔄",
            "blocked": "⚠️",
            "not_started": "⏳"
        }[status]

    def _format_result(self, result: Any) -> str:
        """格式化执行结果"""
        if result is None:
            return "暂无结果"
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
