import multiprocessing
import sys
from io import StringIO
from typing import Dict

from tool.base import BaseTool


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    # 工具名称，用于标识这个工具的唯一ID
    name: str = "python_execute"

    # 工具描述，说明工具的功能和使用注意事项
    # 特别提醒用户只有print输出会被捕获，函数返回值不会被获取
    description: str = "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."

    # 定义工具参数的JSON Schema格式
    # 这里只接受一个必填参数code，类型为字符串
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    # 内部方法，实际执行Python代码的核心逻辑
    # 使用try-except-finally结构确保资源正确释放
    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        # 保存原始标准输出，以便后续恢复
        original_stdout = sys.stdout
        try:
            # 创建StringIO缓冲区来捕获print输出
            output_buffer = StringIO()
            # 重定向标准输出到我们的缓冲区
            sys.stdout = output_buffer
            # 在安全环境下执行代码，使用相同的全局变量字典作为locals和globals
            exec(code, safe_globals, safe_globals)
            # 将缓冲区内容存入结果字典
            result_dict["observation"] = output_buffer.getvalue()
            # 标记执行成功
            result_dict["success"] = True
        except Exception as e:
            # 捕获任何异常，将错误信息存入结果
            result_dict["observation"] = str(e)
            # 标记执行失败
            result_dict["success"] = False
        finally:
            # 无论成功与否，都恢复原始标准输出
            sys.stdout = original_stdout

    # 异步执行方法，提供超时功能
    # 使用多进程来隔离执行环境，确保能安全终止长时间运行的代码
    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """

        # 使用多进程Manager创建共享字典来存储结果
        with multiprocessing.Manager() as manager:
            # 初始化结果字典
            result = manager.dict({"observation": "", "success": False})

            # 创建安全的全局变量环境
            # 处理不同Python版本中__builtins__的不同表现形式
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}

            # 创建子进程来执行代码
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            # 启动子进程
            proc.start()
            # 等待子进程完成，最多等待timeout秒
            proc.join(timeout)

            # 检查进程是否仍在运行（超时情况）
            if proc.is_alive():
                # 终止超时进程
                proc.terminate()
                # 等待进程完全终止
                proc.join(1)
                # 返回超时信息
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }
            # 返回执行结果
            return dict(result)
