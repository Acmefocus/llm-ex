
# ---- core/decorators.py ----
from functools import wraps
from .registry import ToolRegistry
from .security import permission_required
from .validator import validate_parameters
from .cache import cacheable

registry = ToolRegistry()

def tool(name: str, description: str = "", **options):
    """工具装饰器工厂"""
    
    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        
        # 收集元数据
        metadata = {
            'name': name,
            'description': description,
            'parameters': inspect.get_annotations(func),
            'return_type': inspect.get_annotations(func).get('return'),
            'options': options
        }
        
        # 应用功能装饰器
        wrapped = validate_parameters(func)(wrapper)
        wrapped = permission_required(options.get('permissions', []))(wrapped)
        wrapped = cacheable(**options.get('cache', {}))(wrapped)
        
        # 注册工具
        registry.register(name, wrapped, metadata)
        
        return wrapped
    return decorator
