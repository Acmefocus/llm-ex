
# ---- core/validator.py ----
from pydantic import create_model, ValidationError
import inspect

class ValidationError(Exception):
    pass

def validate_parameters(func: Callable):
    """参数验证装饰器"""
    
    model = _create_validation_model(func)
    
    @wraps(func)
    def wrapper(params: dict):
        try:
            validated = model(**params).dict()
            return func(**validated)
        except ValidationError as e:
            raise ValidationError(str(e))
    
    async def async_wrapper(params: dict):
        try:
            validated = model(**params).dict()
            return await func(**validated)
        except ValidationError as e:
            raise ValidationError(str(e))
    
    def _create_validation_model(f):
        fields = {}
        for name, param in inspect.signature(f).parameters.items():
            if name == 'self':
                continue
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = ... if param.default == inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)
        return create_model(f'{f.__name__}Params', **fields)
    
    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
