# Memory/GlobalMemory.py

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, model_validator
from llm import LLM
from Infrastructure.logger import logger
from Infrastructure.schema import Plan, Message
import json
import os
import heapq
import math
import re


class LongTermMemory(BaseModel):
    """长期记忆（知识模型），是 计划成功完成后的经验总结 的数据类型，可从Memory/knowledge.jsonl读取与存储"""
    # 计划经验总结
    title: str = Field(default="", description="计划标题，用于快速识别记忆内容")
    key_words: List[str] = Field(default_factory=list, description="关键词列表，用于快速检索记忆")
    summary: str = Field(default="", description="从执行结果中提取的经验教训和改进建议")
    importance: int = Field(
        default=3, 
        ge=1, 
        le=10, 
        description="重要性评分(1-10)，用于删除、保留记忆优先级排序"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="记忆创建时间，自动记录当前时间"
    )

    @classmethod
    def create_empty(cls):
        """创建一个空的长期记忆对象"""
        return cls(
            title="",
            key_words=[],
            summary="",
            importance=0
        )

    # 格式转换
    def to_string(self) -> str:
        """将记忆对象转换为自然语言格式的字符串
        
        返回格式:
        【{title}】 
        【{key_words}】
        Summary: 
        {summary}
        """
        return (f"【{self.title}】 \n"
                f"【{', '.join(self.key_words)}】\n"
                f"Summary: \n{self.summary}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """将记忆对象转换为字典格式（时间戳会转换为ISO格式）"""
        return {
            "title": self.title,
            "key_words": self.key_words,
            "summary": self.summary,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """将记忆对象转换为JSON字符串
        
        返回格式:
        JSON字符串，包含所有字段
        
        用途:
        - 适合网络传输或持久化存储
        - 完全兼容JSON标准
        - 保留Pydantic的验证能力
        
        特点:
        - 使用Pydantic的model_dump_json方法
        - 包含所有字段，包括默认值和空值
        - 时间戳自动转为ISO格式字符串
        
        与to_dict()的区别:
        - 返回字符串而非字典
        - 保留Pydantic特有的字段验证信息
        """
        return self.model_dump_json()

    #生成文件保存长期记忆
    def save_to_file(self, file_path: str = "Memory/knowledge.jsonl"):
        """将长期记忆添加到Memory/knowledge.jsonl文件中"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(self.to_json() + '\n')

    # 从文件读取长期记忆
    @classmethod
    def load_from_file(cls, file_path: str = "Memory/knowledge.jsonl") -> List['LongTermMemory']:
        """从文件加载长期记忆"""
        long_term_memories = []
        # 文件存在性检查
        if not os.path.exists(file_path):
            logger.info(f"LongTermMemory: 文件不存在, 路径: {file_path}")
            return long_term_memories
        # 文件格式检查
        if not file_path.lower().endswith('.jsonl'):
            logger.warning(f"LongTermMemory: 文件格式可能无效, 预期.jsonl文件, 实际路径: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = 0
                valid_lines = 0
                error_lines = 0
                
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    
                    # 跳过空行
                    if not line:
                        continue
                    
                    try:
                        # 解析JSON行
                        memory_data = json.loads(line)
                        
                        # 验证必要字段
                        required_fields = ['title', 'key_words', 'summary', 'importance']
                        if not all(field in memory_data for field in required_fields):
                            missing_fields = [f for f in required_fields if f not in memory_data]
                            raise ValueError(f"缺失必需字段: {missing_fields}")
                        
                        # 处理时间戳格式
                        if 'timestamp' in memory_data:
                            # 支持多种时间格式
                            if isinstance(memory_data['timestamp'], str):
                                memory_data['timestamp'] = datetime.fromisoformat(memory_data['timestamp'])
                            elif isinstance(memory_data['timestamp'], (int, float)):
                                memory_data['timestamp'] = datetime.fromtimestamp(memory_data['timestamp'])
                        
                        # 验证并创建对象
                        long_term_memories.append(cls(**memory_data))
                        valid_lines += 1
                    
                    except json.JSONDecodeError as e:
                        error_lines += 1
                        logger.error(
                            f"JSON解析错误 行号:{line_num} | 错误:{e} | 内容片段:'{line[:50]}...'"
                        )
                    except ValueError as e:
                        error_lines += 1
                        logger.error(
                            f"数据类型错误 行号:{line_num} | 错误:{e} | 内容片段:'{line[:50]}...'"
                        )
                    except Exception as e:
                        error_lines += 1
                        logger.exception(
                            f"未知错误 行号:{line_num} | 错误类型:{type(e).__name__} | 内容片段:'{line[:50]}...'"
                        )
                
                # 添加加载统计
                if total_lines > 0:
                    success_rate = valid_lines / total_lines * 100
                    logger.info(
                        f"LongTermMemory: 从文件加载完成 | "
                        f"总行数:{total_lines} | 有效行:{valid_lines} | 错误行:{error_lines} | "
                        f"成功率:{success_rate:.1f}%"
                    )
                
                return long_term_memories
                
        except FileNotFoundError:
            logger.warning(f"LongTermMemory: 文件在检查后消失, 路径: {file_path}")
        except PermissionError as e:
            logger.error(f"LongTermMemory: 文件权限错误, 路径: {file_path} | 错误: {e}")
        except OSError as e:
            logger.error(f"LongTermMemory: 系统级IO错误, 路径: {file_path} | 错误: {e}")
        except Exception as e:
            logger.exception(f"LongTermMemory: 加载文件时发生意外错误, 路径: {file_path} | 错误: {type(e).__name__}")
        
        return long_term_memories
# ----------------- PlanContext 核心类 -----------------
class PlanContext(BaseModel):
    class Config:
        """Pydantic配置类"""
        arbitrary_types_allowed = True  # 显式声明任意类型字段的支持

    """跨步骤上下文存储"""
    current_plan_id: str = Field(default="", description="关联计划ID")
    current_step_index: int = Field(default=0)
    step_data: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, str] = Field(
        default_factory=dict,
        description="步骤共享数据（JSON格式字符串存储）"
    )
    llm: LLM = Field(default_factory=lambda: LLM()),  # 语言模型实例

    def __init__(self, **data):
        # 前置处理确保llm有效
        if 'llm' not in data or not isinstance(data.get('llm'), LLM):
            data['llm'] = LLM()
            
        super().__init__(**data)
# ************************************************************
#****添加、更新的方法  在executor完成任务或者达到最大步数时，
# 回顾所有的memory，提取或更新重要的信息****
            # 等到executor完成任务或者达到最大步数时，
            # 提取memory中与用户需求相关的内容作为value
    async def set_step_context(
        self, 
        plan: Plan,
        current_step_index: int, 
        step_result: str,
    ):
        """添加/更新 步骤上下文 - 使用LLM提取跨步骤关键信息
        
        流程：
        1. 提取计划标题、当前步骤描述和执行记忆
        2. 构造LLM提示词，要求提取跨步骤关键信息
        3. 将LLM返回的关键信息存储到上下文
        
        生成键: 步骤索引（从1开始）
        值: 跨步骤关键信息 (JSON格式)
        """
        # 1. 提取当前步骤信息
        current_step_info = plan.steps[current_step_index]

        self.current_step_index = current_step_index
        self.step_data[current_step_index] = current_step_info
        
        # 2. 准备LLM提示词
        system_prompt = Message.system_message(
            "你是智能数据分析整合引擎，专为多步骤Agent系统优化上下文共享。\n"
            "你的核心任务是从步骤执行结果中提取关键信息，以供各步骤间上下文的数据共享。严格遵守以下规则：\n"
            "1. **角色与边界**: \n"
            "   - 仅处理提供的计划执行数据，不依赖外部知识\n"
            "   - 输出必须是结构化JSON，禁止解释性文本\n"
            "2. **语义驱动智能整合数据处理**\n"
            "   - 识别三类关键数据：\n"
            "     ① 资源类（URL/文件）→ 自动主题分组（如'景点参考网址'）\n"
            "     ② 核心事实（数值/决策/结论）→ 直接提取（如'最佳游览时间'）\n"
            "     ③ 矛盾点 → 标记验证源（如'开放时间冲突'）\n"
            "   - 输出规则：\n"
            "     • 同主题URL→换行连接\n"
            "     • 关键事实→保持原样\n"
            "     • 冗余信息→自动过滤\n"
            "3. **风险控制机制**\n"
            f"   - 输出键名必须使用当前步骤实际索引（如'{current_step_index+1}'）\n"
            "   - 状态验证：status≠COMPLETED时返回{\"error\": \"step not completed\"}\n"
            "   - 空值处理：无可提取数据时返回空字典\n"
            "   - 格式容错：自动处理JSON/文本混合输入\n"
            "4. **技术参数**: \n"
            "   - temperature=0.2（低值确保确定性和可复现性)。\n"
        )
        user_prompt = Message.user_message(
            "分析以下步骤执行信息，提取对后续步骤关键的数据。\n"
            "输入数据详情：\n"
            "当前跨步骤上下文：\n"
            f"{self.get_step_context()}\n"
            "计划全貌：\n"
            f"{self._format_plan_steps(plan)}\n"
            "当前步骤信息：\n"
            f"- **当前步骤**: {current_step_index+1}\n"
            f"  - `description`: \"{current_step_info.description}\"\n"
            f"  - `expected_output`: \"{current_step_info.expected_output}\"\n"
            f"  - `status`: \"{current_step_info.status}\"  约束条件：必须为COMPLETED，否则返回{{\"error\": \"step not completed\", \"key_data\": []}}\n"
            f"- **步骤执行结果(step_result)**: \"{step_result}\"\n"
            "————————————————————————————————————————————————————————\n"
            "数据处理要求:\n"
            "1. 只提取关键资源（文件/代码/数据……）和核心决策事实（数值/结论……）\n"
            "2. 按智能主题合并：\n"
            "   - 同类资源→列表存储\n"
            "   - 关键事实→独立字段\n"
            "3. 输出结构：\n"
            f"   {{\"{current_step_index+1}\": {{\n"
            "        \"主题名称\": \"数据内容\"\n"
            "   }}}\n"
            "4. 错误处理：\n"
            "   - 数据冲突时：return {\"error\": \"data conflict\"}\n"
            "   - 输入无效时：return {\"error\": \"invalid input\"}\n"
            "**输出示例**\n"
            "正确案例：{\n"
            f"   {{\"5\": {{\"政策指导要点\":\"1.推行电子票务系统\n2.延长旺季开放时间\", \"均价波动\":\"旺季+35%\"}}}}\n"
            "错误案例：{\"error\": \"step not completed\"}"
        )

        # 3. 调用LLM提取关键信息
        try:

            response = await self.llm.ask(
                messages=[user_prompt],
                system_msgs=[system_prompt]
            )
            
            # 键：步骤索引（从1开始）
            key = current_step_index + 1
            
            # 4. 存储到上下文
                    # 生成键: 步骤索引（从1开始）
                    # 值: 跨步骤关键信息 (JSON格式)
            self.data[key] = response
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"提取plancontext时LLM响应格式错误: {e}")
            # === 提供默认值避免崩溃 ===
            self.data[key] = {"error": "llm_response_error", "key_data": []}
        except AttributeError as e:
            # === 处理llm缺失问题 ===
            logger.error(f"LLM实例不可用: {str(e)}")
            self.data[key] = {"error": "llm_unavailable", "key_data": []}
        except Exception as e:
            # === 通用错误处理 ===
            logger.exception(f"提取上下文时发生意外错误: {str(e)}")
            self.data[key] = {"error": "unexpected_error", "key_data": []}
    
    # set_step_context的辅助方法，读取计划的步骤信息并格式化
    def _format_plan_steps(self, plan: Plan) -> str:
        """
        将计划的步骤信息格式化为易于LLM理解的自然语言文本
        
        参数:
            plan: Plan对象, 包含步骤列表
            
        返回:
            格式化的步骤描述字符串
        """
        if not plan.steps:
            return f"计划 '{plan.title}' 尚未包含任何步骤"
        
        output = [f"## 计划: {plan.title} (ID: {plan.plan_id})"]
        
        for i, step in enumerate(plan.steps, 1):
            step_info = [
                f"### 步骤 {i}:",
                f"描述: {step.description}",
            ]
            
            # 添加预期输出描述
            if step.expected_output:
                step_info.append(f"预期输出: {step.expected_output}")
            
            # 添加当前状态
            step_info.append(f"状态: {step.status}")
            output.append("\n".join(step_info))
        
        return "\n\n".join(output)

    def get_step_context(self, step_idx: Optional[int] = None) -> str:
        """
        获取步骤上下文数据，格式化为LLM可读的文本
        
        参数:
            step_idx (可选): 1-base步骤索引。如未提供，则返回所有步骤的文本
        返回:
            - 当step_idx提供时: 该步骤关键数据的格式化文本
            - 当step_idx未提供时: 所有步骤关键数据的格式化文本（单个字符串）
        """
        def format_step_data(step_index: int, data: Any) -> str:
            """将单步骤数据格式化为LLM可读的文本"""
            # 如果是字符串尝试解析
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return f"步骤 {step_index} - 原始文本: {data[:100]}..."
            
            # 检查错误响应
            if isinstance(data, dict) and "error" in data:
                return f"步骤 {step_index} 错误: {data['error']}\n"
            
            # 查找包含关键数据的键（适配不同LLM返回格式）
            key_data = None
            if isinstance(data, dict):
                # 尝试各种可能的键名
                for key in [str(step_index), f"step_{step_index}", "key_data", "extracted_data"]:
                    if key in data and isinstance(data[key], list):
                        key_data = data[key]
                        break
            
            # 没有找到有效数据
            if not key_data:
                return f"步骤 {step_index} - 无关键数据\n\n"
                
            # 格式化文本输出
            lines = [f"## 步骤 {step_index} 关键数据:"]
            for i, item in enumerate(key_data, 1):
                if not isinstance(item, dict):
                    continue
                    
                key_val = item.get('key', '未知')
                value = item.get('value', '未知')
                relevance = item.get('relevance', '未知')
                lines.append(f"- **{key_val}**: {value}")
                lines.append(f"  用途: {relevance}\n")
            
            return "\n".join(lines) + "\n\n"

        # 获取特定步骤上下文
        if step_idx is not None:
            data = self.data.get(step_idx)
            if not data:
                return f"步骤 {step_idx} 无上下文数据"
            return format_step_data(step_idx, data)
        
        # 获取所有步骤上下文，整合为单个文本
        all_steps_text = []
        keys = sorted(k for k in self.data.keys() if isinstance(k, int))
        
        for step_index in keys:
            data = self.data[step_index]
            step_text = format_step_data(step_index, data)
            all_steps_text.append(step_text)
        
        # 汇总
        if all_steps_text:
            return "\n".join(all_steps_text)
        return "无可用上下文数据"

    def to_text(self) -> str:
        """转换为自然语言文本，适合LLM输入"""
        text = f"计划上下文 (ID: {self.current_plan_id}):\n"
        for key, context_str in self.data.items():
            # 提取步骤ID
            step_id = key.replace("step_", "")
            
            try:
                context_data = json.loads(context_str)
                text += f"步骤 {step_id}:\n"
                
                # 格式化各种类型的值
                for k, v in context_data.items():
                    if isinstance(v, dict):
                        # 字典类型转为字符串表示
                        text += f"  - {k}: {json.dumps(v, ensure_ascii=False)}\n"
                    elif isinstance(v, list):
                        # 列表类型转为字符串列表
                        items = "\n".join([f"    * {item}" for item in v])
                        text += f"  - {k}:\n{items}\n"
                    else:
                        # 简单类型直接显示
                        text += f"  - {k}: {v}\n"
            except json.JSONDecodeError:
                # 非JSON格式的数据保持原样显示
                text += f"步骤 {step_id}: {context_str}\n"
        return text
    
    def clear(self):
        """清除整个上下文"""
        self.data.clear()
        self.current_plan_id = ""

# ----------------- GlobalMemory 核心类 -----------------
class GlobalMemory:
    """
    全局记忆管理系统，包括长期记忆和跨步骤上下文
    功能：
    - 长期记忆管理：
        - 从文件加载长期记忆
        - 保存新的长期记忆，并根据重要性进行管理删除
        - 检索相关的经验
    - 跨步骤上下文管理：
        - 跨步骤共享数据
        - 存储和获取跨步骤上下文
    
    单例模式实现：确保整个应用只有一个 GlobalMemory 实例
    """
    # 单例实例引用
    _instance = None
    
    def __new__(cls):
        """
        单例模式实现
        - 确保整个应用生命周期中只有一个 GlobalMemory 实例
        - 首次调用时创建实例
        - 后续调用返回同一实例
        """
        if cls._instance is None:
            # 创建新实例
            cls._instance = super().__new__(cls)
            # 设置初始化标记
            cls._instance.__initialized = False
        return cls._instance
    
    def __init__(
        self, 
        llm: Optional[LLM] = None,
        experience_len: int = 3,
        max_memories: int = 50, 
        importance_decay_factor: float = 0.95
    ):
        """
        初始化全局记忆系统
        - 只会在首次创建单例时执行
        - 后续调用返回同一实例，不会重新初始化
        
        参数:
        - llm: LLM实例，用于语言模型操作。如未提供，创建默认实例
        - experience_len: 经验列表最大长度 (1-20)
        - max_memories: 最大记忆存储数量 (≤100)
        - importance_decay_factor: 重要性衰减系数 (≤1.0)
        """
        # 检查是否已初始化
        if self.__initialized:
            return
            
        # 处理 LLM 默认值
        if llm is None:
            llm = LLM()  # 创建默认LLM实例
        
        # 核心属性 ---------------------------------------------------
        self.llm = llm
        """语言模型实例，用于生成摘要和关键信息提取"""
        
        self.experience_len = max(1, min(experience_len, 20))
        """经验列表最大长度，限制为1-20之间的值（性能优化）"""
        
        self.max_memories = min(max_memories, 100)
        """文件存储中最多保存的记忆数量，限制≤100"""
        
        self.importance_decay_factor = min(importance_decay_factor, 1.0)
        """重要性得分随时间衰减的系数 (0.0-1.0)，影响记忆保留策略"""
        
        # 动态数据存储 ---------------------------------------------------
        self.plans = {}
        """
        计划表字典
        - 键: 计划ID (str)
        - 值: Plan对象
        用途：存储当前所有活动计划的详细信息
        """
        
        self.experience = []
        """
        经验列表
        - 类型: List[LongTermMemory]
        用途：存储从长期记忆文件中加载的当前最相关经验
        长度限制: 由experience_len控制
        """
        
        self.plan_contexts = {}
        """
        跨步骤共享数据字典
        - 键: 计划ID (str)
        - 值: PlanContext对象
        用途：存储每个计划跨步骤共享的上下文数据
        """
        
        # 标记已初始化 - 防止重复初始化
        self.__initialized = True

# 那创建的时候就不能在工具中直接创建，不然会被删除的，需要在流程中创建这个对象，最后在流程中删除
    def sync_plans(self, plans: Dict[str, Plan]):
        """同步最新计划表数据（在计划创建后调用）"""
        self.plans = plans
        
        # 为新计划创建上下文
        for plan_id in self.plans:
            if plan_id not in self.plan_contexts:
                self.plan_contexts[plan_id] = PlanContext(current_plan_id=plan_id)

    def sync_plan(self, plan: Plan):
        """同步单个计划数据（在计划变更后调用）
        
        参数:
            plan: Plan对象，包含计划信息
        逻辑:
            - 如果plan_id已存在，则更新对应计划
            - 如果plan_id不存在，则添加新计划
            - 同时维护对应的PlanContext
        """
        # 更新或添加计划
        self.plans[plan.plan_id] = plan
        
        # 如果该计划没有上下文，则创建新的上下文
        if plan.plan_id not in self.plan_contexts:
            self.plan_contexts[plan.plan_id] = PlanContext(current_plan_id=plan.plan_id)
        
    # 根据用户请求检索相关经验
    def retrieve_relevant_experience(self, request: str):
        """根据加权因素的评分排序检索相关经验"""
        # 使用最小堆获取top-k相关经验
        scored_memories = []
        
        try:
            # 尝试加载经验文件
            all_experience = LongTermMemory.load_from_file("Memory/knowledge.jsonl")
            
            # 如果经验不足，使用空列表
            if not all_experience:
                logger.info("没有可用的长期记忆经验，将使用空经验列表")
                self.experience = []
                return
                
            # 计算每个记忆的相关性得分
            for memory in all_experience:
                score = self._calculate_relevance_score(request, memory)
                heapq.heappush(scored_memories, (score, memory))
                
                # 维持堆大小
                if len(scored_memories) > self.experience_len:
                    heapq.heappop(scored_memories)
                    
            # 返回高相关性降序排序结果
            self.experience = [mem for _, mem in sorted(scored_memories, reverse=True)]
            
            # 如果最终经验数量不足，补充空记忆
            while len(self.experience) < self.experience_len:
                self.experience.append(LongTermMemory.create_empty())
                
        except Exception as e:
            logger.error(f"检索knowledge.jsonl时出错: {e}")
            # 出错时返回空经验列表
            self.experience = [LongTermMemory.create_empty() for _ in range(self.experience_len)]

    # 检索knowledge的辅助方法，计算相关性得分
    def _calculate_relevance_score(self, request: str, memory: LongTermMemory) -> float:
        """
        计算用户请求(request)与长期记忆(memory)之间的相关性得分
        分数范围0-1，值越大表示相关性越强
        
        参数:
            request: 用户的请求文本
            memory: LongTermMemory对象，存储长期记忆信息
        返回:
            归一化后的相关性分数
        """
        # 1. 辅助函数
        # 2. 计算各项分数（统一到0-1）
        # -------------------------------------
        
        # 标题语义相似度（基于词重叠）
        title_sim = self.__simple_title_similarity(request, memory.title)
        
        # 关键词匹配度（含位置权重）
        keyword_match = self.__keyword_match_score(request, memory.key_words)
        
        # 记忆新鲜度（时间衰减因子）
        time_factor = self.__time_decay_factor(memory.timestamp)
        
        # 重要性评分
        importance = self.__normalize_importance(memory.importance)
        
        # 3. 加权求和（权重和为1）
        # -------------------------------------
        weights = {
            'title': 0.4,      # 最大权重：标题相似最重要
            'keywords': 0.3,   # 关键词匹配次重要
            'time': 0.1,       # 时间影响较小
            'importance': 0.2  # 用户标注的重要性
        }
        
        # 最终相关性分数 = Σ(子分数×权重)
        return (
            weights['title'] * title_sim +
            weights['keywords'] * keyword_match +
            weights['time'] * time_factor +
            weights['importance'] * importance
        )
    
    # 1. 辅助函数  统一量纲到0-1范围
    def __simple_title_similarity(self, text1, text2):
        """
        计算两个标题的简化Jaccard相似度
        基本思想：将标题视为词集合，计算交集比例
        """
        # 转换为小写并分词（提高匹配容错性）
        words1 = set(text1.lower().split())  
        words2 = set(text2.lower().split())
        
        # 空集保护（两者皆空时返回0而非出错）
        if not words1 and not words2:
            return 0.0  
            
        # 计算Jaccard系数 = 交集/并集
        intersection = words1 & words2
        return len(intersection) / (len(words1) + len(words2) - len(intersection))
    
    def __keyword_match_score(self, plan_title, memory_keywords):
        """
        计算计划标题与记忆关键词的匹配度（带权重）
        特点：
        1. 支持部分匹配（如"api"匹配"api_key"）
        2. 关键词有序，前位词权重更高
        3. 组合简单匹配+加权匹配分数
        """
        # 提取标题中的有效单词（长度≥3字符）
        plan_words = re.findall(r'\w{3,}', plan_title.lower())  
        
        # 边界检查：空词集返回0分
        if not memory_keywords or not plan_words:
            return 0.0
            
        # A. 基本匹配度计算
        match_count = 0
        for keyword in memory_keywords:
            kw = keyword.lower()
            # 双向部分匹配：关键词在标题词中或标题词在关键词中
            if any(kw in word for word in plan_words) or any(word in kw for word in plan_words):
                match_count += 1
                
        # B. 加权匹配度计算（考虑词序重要性）
        weighted_match = 0.0
        for i, keyword in enumerate(memory_keywords):
            # 权重衰减：第1个词权重1.0，第2个0.5，第3个0.33...
            weight = 1.0 / (i + 1)  
            kw = keyword.lower()
            if any(kw in word for word in plan_words):
                weighted_match += weight
        
        # C. 归一化处理
        simple_match = match_count / len(memory_keywords)  # 匹配率
        
        # 计算最大可能权重和（用于归一化）
        max_weight_sum = sum(1.0/(idx+1) for idx in range(len(memory_keywords)))
        normalized_weighted_match = weighted_match / max_weight_sum  # 加权匹配率
        
        # D. 最终分数 = 简单匹配和加权匹配的平均值
        return (simple_match + normalized_weighted_match) / 2
    
    def __time_decay_factor(self, memory_timestamp):
        """
        计算基于记忆时间的时间衰减因子
        独特设计：使用指数为0.3的幂次衰减（而非线性衰减）
        特点：
        1. 对近期记忆衰减缓慢（前几天的记忆不会大幅降低）
        2. 对远期记忆保留少量影响（非零值）
        公式：1 / (1 + days_old^0.3)
        """
        days_old = (datetime.now() - memory_timestamp).days
        return 1 / (1 + math.pow(days_old, 0.3))  # 平滑衰减曲线
    
    def __normalize_importance(self, importance):
        """
        归一化重要性的实用函数
        核心：将10点制（0-10）线性映射到0-1区间
        防护措施：限制在有效范围避免越界
        """
        return min(1.0, max(0.0, importance / 10))
        
    async def clear_global_memory(self):
        """尝试创建长期记忆，清除全局内存，在流程结束时调用"""
        # 尝试为所有已完成计划创建记忆
        for plan_id in list(self.plans.keys()):
            await self._create_memory_for_completed_plan(plan_id)
            
        # 删除文件中已经保存但不重要的旧记忆
        self._apply_retention_policy()
        
        # 清空内存中的状态
        self.plans.clear()
        for context in self.plan_contexts.values():
            context.clear()
        self.plan_contexts.clear()
    

    async def _create_memory_for_completed_plan(self, plan_id: str):
        """从完成的计划中创建长期记忆"""
        if plan_id not in self.plans:
            return   
        plan = self.plans[plan_id]
        if not hasattr(plan, 'execution_log') or not plan.execution_log:
            return
        # 检查计划完成标记
        if "✅ SUCCESS-COMPLETED ✅" not in plan.execution_log:
            return
        # 总结完成的计划execution_log并生成关键词，创建长期记忆对象
        try:
            # 使用LLM生成高质量的经验总结和重要性评分
            system_message = Message.system_message(
                "你是一名专业的经验萃取师，负责从计划执行日志中提取高价值经验教训。遵循以下规则：\n"
                "1. **核心角色**：\n"
                "   - 专注分析执行日志中可复用的经验和改进点\n"
                "   - 总结的经验需要对未来计划的创建有帮助\n"
                "2. **总结原则**：\n"
                "   - **前瞻性**：聚焦对未来计划创建有直接启发的洞见\n"
                "3. **评分标准**：\n"
                "   - 重要性评分（0-10分）基于：\n"
                "     ✅ 经验复用广度（影响多少未来计划）\n"
                "     ✅ 问题解决深度（避免重大失误的程度）\n"
                "     ✅ 创新价值（带来方法论突破）\n"
                "4. **关键词规范**：\n"
                "   - 每个关键词为2-4字的名词短语（如\"API限流\"）\n"
                "   - 至少包含1个问题类和1个解决方案类关键词\n"
                "5. **输出约束**：\n"
                "   - 强制JSON格式，包含summary/keywords/importance\n"
                "   - 重要性保留1位小数（如7.5）"
            )
            user_message = Message.user_message(
                "根据以下计划执行日志萃取高价值经验并进行重要性评分:\n"
                f"计划标题: {plan.title}\n"
                f"执行日志:\n{plan.execution_log}\n\n"
                "请按维度分析日志得到summary：\n"
                "1. **卡顿归因**：导致卡顿的计划设置或步骤操作\n"
                "2. **成功要素**：可复用的最佳实践\n"
                "3. **优化杠杆**：需系统化改进的环节\n\n"
                "请按维度分析日志得到keywords：\n"
                "1. **需求**：用户需求是什么方面的\n"
                "2. **流程**：处理需求的关键流程、思考方向\n"
                "请按维度分析日志得到importance：\n"
                "1. **广度**：对未来计划创建的影响\n"
                "2. **深度**：解决问题的深度\n"
                "3. **创新**：带来的方法论突破\n\n"
                "请以JSON格式返回: {\n"
                "  \"summary\": \"经验总结内容\",\n"
                "  \"keywords\": [\"关键词1\", \"关键词2\"], \n" 
                "  \"importance\": 1-10之间的一位小数\n"
                "}"
            )
            response = await self.llm.ask(
                messages=[user_message], 
                system_msgs=[system_message]
            )
            # 添加JSON格式验证
            if not response.startswith("{"):
                response = "{" + response.split("{", 1)[-1]  # 尝试修复JSON格式
                if not response.endswith("}"):
                    response += "}"
            # 解析LLM的响应
            try:
                response = json.loads(response)
                summary = response.get("summary", "")
                keywords = response.get("keywords", [])
                importance_score = response.get("importance", 4)
            except json.JSONDecodeError:
                logger.error("LLM响应格式错误，无法解析。")

            memory = LongTermMemory(
                title=plan.title,
                key_words=keywords,
                summary=summary,
                importance=importance_score
            )
            logger.info(f"成功创建长期记忆: {memory.title}")
                        
            # 所有的内容都保存到同一个jsonl文件中
            memory.save_to_file("Memory/knowledge.jsonl")
            logger.info(f"成功保存长期记忆: {memory.title}")
        except (json.JSONDecodeError, KeyError) as e:
            # 添加更详细的错误处理
            logger.error(f"LLM响应格式错误: {e}")
            # 使用默认值创建记忆
            memory = LongTermMemory(
                title=plan.title,
                key_words=[], 
                summary="",
                importance=3
            )
            memory.save_to_file(f"Memory/knowledge.jsonl")

    def _apply_retention_policy(self, file_path: str = "Memory/knowledge.jsonl"):
        """应用保留策略，保证文件中的记忆不超过最大数量"""
        # 加载当前所有记忆
        all_memories = LongTermMemory.load_from_file(file_path)
        
        # 如果不超过最大数量，直接返回
        if len(all_memories) <= self.max_memories:
            return
            
        # 计算每个记忆的时效性得分
        current_time = datetime.now()
        scores = []
        for memory in all_memories:
            # 时间衰减因子
            time_factor = max(0.1, 1 - (current_time - memory.timestamp).days / 365)
            
            # 综合得分 = 重要性 * 时间因子 * 衰减系数
            score = memory.importance * time_factor * self.importance_decay_factor
            scores.append((score, memory))
        
        # 使用堆获取最高得分的 max_memories 个记忆
        top_memories = heapq.nlargest(
            self.max_memories, 
            scores, 
            key=lambda x: x[0]
        )
        
        # 只保留这些记忆
        top_memory_list = [memory for _, memory in top_memories]
        
        # 重写文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for memory in top_memory_list:
                f.write(memory.to_json() + '\n')
