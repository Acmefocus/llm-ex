
# ---- core/executor.py ----
from .registry import ToolRegistry
from .exceptions import ToolNotFound, ExecutionFailed
from ..utils.logger import ToolLogger

class ToolExecutor:
    """工具执行引擎"""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.logger = ToolLogger()
    
    async def execute(self, tool_name: str, params: dict, context: dict):
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ToolNotFound(f"Tool {tool_name} not registered")
        
        try:
            self.logger.log_invocation(tool_name, params)
            
            # 执行工具
            if tool['is_async']:
                result = await tool['func'](context, **params)
            else:
                result = tool['func'](context, **params)
                
            self.logger.log_success(tool_name)
            return result
            
        except Exception as e:
            self.logger.log_failure(tool_name, str(e))
            raise ExecutionFailed(f"Tool execution failed: {str(e)}") from e
