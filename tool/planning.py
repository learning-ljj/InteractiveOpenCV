# tool/planning.py
from typing import Dict, List, Literal, Optional

from Infrastructure.exceptions import ToolError
from tool.base import BaseTool, ToolResult


_PLANNING_TOOL_DESCRIPTION = """
A planning tool that allows the agent to create and manage plans for solving complex tasks.
The tool provides functionality for creating plans, updating plan steps, and tracking progress.
"""


class PlanningTool(BaseTool):
    """
    规划工具类，用于创建和管理多步骤任务的执行计划
    """
    
    # 工具元数据定义
    name: str = "planning"  # 工具名称，用于系统识别
    description: str = _PLANNING_TOOL_DESCRIPTION  # 工具功能描述
    
    # OpenAPI规范的参数定义 (用于LLM工具调用)
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "The command to execute. Available commands: create, update, list, get, set_active, mark_step, delete.",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "Unique identifier for the plan. Required for create, update, set_active, and delete commands. Optional for get and mark_step (uses active plan if not specified).",
                "type": "string",
            },
            "title": {
                "description": "Title for the plan. Required for create command, optional for update command.",
                "type": "string",
            },
            "steps": {
                "description": "List of plan steps. Required for create command, optional for update command.",
                "type": "array",
                "items": {"type": "string"},
            },
            "step_index": {
                "description": "Index of the step to update (0-based). Required for mark_step command.",
                "type": "integer",
            },
            "step_status": {
                "description": "Status to set for a step. Used with mark_step command.",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
            },
            "step_notes": {
                "description": "Additional notes for a step. Optional for mark_step command.",
                "type": "string",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    # 计划存储结构
    plans: dict = {}  # 计划仓库，格式为字典: {plan_id: plan_data}
    _current_plan_id: Optional[str] = None  # 当前活动计划ID，用于简化操作

    async def execute(
        self,
        *,
        command: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[
            Literal["not_started", "in_progress", "completed", "blocked"]
        ] = None,
        step_notes: Optional[str] = None,
        **kwargs,
    ):
        """
        执行规划工具命令
        
        参数:
            command: 要执行的操作
            plan_id: 计划的唯一标识符
            title: 计划标题
            steps: 计划步骤列表
            step_index: 步骤索引
            step_status: 步骤状态
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

    def _create_plan(self, plan_id: str, title: str, steps: List[str]) -> ToolResult:
        """
        创建新计划内部实现
        逻辑流程:
        1. 参数校验 → 2. 初始化数据结构 → 3. 存储计划 → 4. 设为活动计划
        """
        # 参数有效性检查
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
            or not all(isinstance(step, str) for step in steps)
        ):
            raise ToolError(
                "create命令需要steps参数，且必须是非空字符串列表"
            )

        # 创建新计划并初始化步骤状态
        plan = {
            "plan_id": plan_id,  # 计划唯一标识
            "title": title,      # 人类可读标题
            "steps": steps,      # 步骤文本列表
            "step_statuses": ["not_started"] * len(steps),  # 初始化所有步骤为未开始
            "step_notes": [""] * len(steps)  # 初始化空备注
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # Set as active plan

        return ToolResult(
            output=f"计划创建成功，ID: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(self, plan_id: str, title: Optional[str], steps: Optional[List[str]]) -> ToolResult:
        """
        更新计划实现要点:
        - 标题更新: 直接替换
        - 步骤更新: 智能合并状态和备注
           - 位置相同的未修改步骤保留原状态
           - 新增/修改的步骤重置为未开始状态
        """
        if not plan_id:
            raise ToolError("update命令需要plan_id参数")

        if plan_id not in self.plans:
            raise ToolError(f"找不到ID为 {plan_id} 的计划")

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise ToolError(
                    "update命令的steps参数必须是字符串列表"
                )

            # 保留未修改步骤的状态和备注
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # 创建新的状态和备注列表
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # 如果步骤在相同位置且未修改，保留原状态和备注
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

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
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} 步骤完成"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        """获取特定计划的详细信息"""
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
            output=f"计划 '{plan_id}' 已设为活动计划\n\n{self._format_plan(self.plans[plan_id])}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        """标记步骤状态"""
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

        # 边界索引检查，step_index是否有效
        if step_index < 0 or step_index >= len(plan["steps"]):
            raise ToolError(
                f"无效的step_index: {step_index}。有效范围: 0 到 {len(plan['steps'])-1}"
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            raise ToolError(
                f"无效的step_status: {step_status}。有效状态: not_started, in_progress, completed, blocked"
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status
            
        # 备注更新逻辑    
        if step_notes:
            plan["step_notes"][step_index] = step_notes

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
        """格式化计划信息用于显示"""
        output = f"计划: {plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # 计算进度统计
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"进度: {completed}/{total_steps} 步骤完成 "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"状态: {completed} 完成, {in_progress} 进行中, {blocked} 阻塞, {not_started} 未开始\n\n"
        output += "步骤:\n"

        # 添加每个步骤及其状态和备注
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   备注: {notes}\n"

        return output
