
# ---- core/registry.py ----
from typing import Dict, Callable, Any
import inspect

class ToolRegistry:
    """工具注册中心"""
    
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, name: str, func: Callable, metadata: dict) -> None:
        self._tools[name] = {
            'func': func,
            'is_async': inspect.iscoroutinefunction(func),
            'metadata': metadata
        }

    def get_tool(self, name: str) -> Dict[str, Any]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, dict]:
        return {name: data['metadata'] for name, data in self._tools.items()}
