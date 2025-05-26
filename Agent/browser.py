import json
from typing import TYPE_CHECKING, Optional

from pydantic import Field, model_validator

from Agent.ToolCall import ToolCallAgent
from Infrastructure.logger import logger
from prompt.browser import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from Infrastructure.schema import Message, ToolChoice
from tool import BrowserUseTool, Terminate, ToolCollection

# 避免循环导入
if TYPE_CHECKING:
    from Agent.Base import BaseAgent  # 仅在类型检查时导入


class BrowserContextHelper:
    """浏览器上下文助手类，用于管理浏览器状态和交互"""
    
    def __init__(self, agent: "BaseAgent"):
        """初始化浏览器上下文助手
        
        参数:
            agent: 关联的基础代理实例
        """
        self.agent = agent
        self._current_base64_image: Optional[str] = None  # 当前浏览器截图缓存
        self._initialized = False  # 初始化状态标记
    
    async def ensure_initialized(self):
        """确保浏览器上下文已初始化"""
        if not self._initialized:
            browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
            if browser_tool and hasattr(browser_tool, "initialize"):
                await browser_tool.initialize()
                self._initialized = True
                logger.debug("浏览器上下文已初始化")
            else:
                logger.warning("浏览器工具不可用或缺少初始化方法")

    async def get_browser_state(self) -> Optional[dict]:
        """获取当前浏览器状态
        
        返回:
            Optional[dict]: 浏览器状态字典或None(获取失败时)
        """
        # 获取浏览器工具实例
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            logger.warning("未找到浏览器工具或工具不支持获取状态")
            return None
            
        try:
            # 调用工具获取状态
            result = await browser_tool.get_current_state()
            if result.error:
                logger.debug(f"浏览器状态错误: {result.error}")
                return None
                
            # 缓存截图数据
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image
            else:
                self._current_base64_image = None
                
            return json.loads(result.output)
        except Exception as e:
            logger.debug(f"获取浏览器状态失败: {str(e)}")
            return None

    async def format_next_step_prompt(self) -> str:
        """格式化浏览器操作提示词
        
        整合当前浏览器状态信息到提示模板中
        """
        browser_state = await self.get_browser_state()
        url_info, tabs_info = "", ""
        content_above_info, content_below_info = "", ""
        results_info = ""  # 预留结果信息位置

        if browser_state and not browser_state.get("error"):
            # 格式化URL和标题信息
            url_info = f"\n   当前URL: {browser_state.get('url', '未知')}\n   页面标题: {browser_state.get('title', '未知')}"
            
            # 处理标签页信息
            tabs = browser_state.get("tabs", [])
            if tabs:
                tabs_info = f"\n   共有{len(tabs)}个标签页"
                
            # 处理滚动位置信息
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)
            if pixels_above > 0:
                content_above_info = f" (上方有{pixels_above}像素内容)"
            if pixels_below > 0:
                content_below_info = f" (下方有{pixels_below}像素内容)"

            # 添加截图到消息历史
            if self._current_base64_image:
                image_message = Message.user_message(
                    content="当前浏览器截图:",
                    base64_image=self._current_base64_image,
                )
                self.agent.memory.add_message(image_message)
                self._current_base64_image = None  # 使用后清空缓存

        # 填充提示模板
        return NEXT_STEP_PROMPT.format(
            url_placeholder=url_info,
            tabs_placeholder=tabs_info,
            content_above_placeholder=content_above_info,
            content_below_placeholder=content_below_info,
            results_placeholder=results_info,
        )

    async def cleanup_browser(self):
        """清理浏览器资源"""
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if browser_tool and hasattr(browser_tool, "cleanup"):
            await browser_tool.cleanup()


class BrowserAgent(ToolCallAgent):
    """浏览器控制代理
    
    功能:
        - 网页导航
        - 元素交互
        - 表单填写
        - 内容提取
    """

    # 代理基本信息
    name: str = "browser"
    description: str = "用于控制浏览器完成任务的代理"

    # 提示模板配置
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 执行限制参数
    max_observe: int = 10000  # 最大观察长度
    max_steps: int = 20  # 最大执行步数

    # 可用工具配置
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(BrowserUseTool(), Terminate())
    )

    # 工具选择策略
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # 浏览器上下文助手
    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "BrowserAgent":
        """初始化浏览器上下文助手"""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """决策下一步操作
        
        集成浏览器状态信息到决策过程中
        """
        self.next_step_prompt = (
            await self.browser_context_helper.format_next_step_prompt()
        )
        return await super().think()

    async def cleanup(self):
        """清理代理资源"""
        await self.browser_context_helper.cleanup_browser()
