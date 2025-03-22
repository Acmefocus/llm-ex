# ---- core/cache.py ----
import hashlib
import json
from datetime import datetime, timedelta

class CacheStore:
    """缓存存储抽象层"""
    
    def __init__(self):
        self._store = {}
    
    def get(self, key: str):
        entry = self._store.get(key)
        if entry and datetime.now() < entry['expiry']:
            return entry['value']
        return None
    
    def set(self, key: str, value: Any, ttl: int):
        self._store[key] = {
            'value': value,
            'expiry': datetime.now() + timedelta(seconds=ttl)
        }

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, store: CacheStore = None):
        self.store = store or CacheStore()
    
    def generate_key(self, func_name: str, params: dict) -> str:
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return f"{func_name}:{param_hash}"

def cacheable(ttl: int = 300, key_strategy: str = 'params'):
    """缓存装饰器实现"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(params: dict):
            mgr = CacheManager()
            cache_key = mgr.generate_key(func.__name__, params)
            cached = mgr.store.get(cache_key)
            if cached:
                return cached
            result = func(params)
            mgr.store.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        async def async_wrapper(params: dict):
            mgr = CacheManager()
            cache_key = mgr.generate_key(func.__name__, params)
            cached = mgr.store.get(cache_key)
            if cached:
                return cached
            result = await func(params)
            mgr.store.set(cache_key, result, ttl)
            return result
        
        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
    return decorator
