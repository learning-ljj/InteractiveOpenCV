from typing import Dict, List, Optional
from pydantic import Field, model_validator

# 导入各种工具和辅助类
from Agent.browser import BrowserContextHelper  # 浏览器上下文助手
from Agent.ToolCall import ToolCallAgent  # 工具调用代理基类
from Infrastructure.config import config  # 应用配置
from Infrastructure.logger import logger  # 日志记录器
from prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT  # 提示模板
from tool import Terminate, ToolCollection  # 工具集合和终止工具
from tool.ask_human import AskHuman  # 询问人类工具
from tool.browser_use_tool import BrowserUseTool  # 浏览器使用工具
from tool.mcp import MCPClients, MCPClientTool  # MCP客户端相关
from tool.python_execute import PythonExecute  # Python执行工具
from tool.str_replace_editor import StrReplaceEditor  # 字符串替换编辑器工具
from tool.web_search import WebSearch  # 网络搜索工具


class Manus(ToolCallAgent):
    """一个多功能通用代理，支持本地和MCP工具"""
    
    # 代理基本信息
    name: str = "Manus"
    description: str = "一个多功能代理，可以使用包括MCP工具在内的多种工具解决各种任务"

    # 提示模板配置
    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 执行限制参数
    max_observe: int = 10000  # 最大观察结果长度
    max_steps: int = 20  # 最大执行步数

    # MCP客户端配置，用于远程工具访问
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # 可用工具集合初始化
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),  # Python执行工具
            BrowserUseTool(),  # 浏览器工具
            WebSearch(),  # 网络搜索工具
            StrReplaceEditor(),  # 字符串替换工具
            AskHuman(),  # 询问人类工具
            Terminate(),  # 终止工具
        )
    )

    # 特殊工具名称列表
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    
    # 浏览器上下文助手
    browser_context_helper: Optional[BrowserContextHelper] = None

    # 已连接的MCP服务器跟踪
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # 服务器ID -> URL/命令
    
    # 初始化状态标记
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """同步初始化基本组件"""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """工厂方法，创建并正确初始化Manus实例"""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """初始化与配置的MCP服务器的连接"""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"已连接到MCP服务器 {server_id} 地址: {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"已连接到MCP服务器 {server_id} 使用命令: {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"连接MCP服务器 {server_id} 失败: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """连接到MCP服务器并添加其工具"""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # 仅添加来自此服务器的新工具
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """断开与MCP服务器的连接并移除其工具"""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # 重建可用工具集合，不包括已断开服务器的工具
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """清理Manus代理资源"""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # 仅在已初始化时断开所有MCP服务器连接
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False
        await super().cleanup()  # 调用父类清理方法

    async def think(self) -> bool:
        """处理当前状态并决定下一步操作"""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        
        # 检查是否正在使用浏览器工具
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # 恢复原始提示
        self.next_step_prompt = original_prompt

        return result
