# ---- core/security.py ----
from functools import wraps
from .exceptions import PermissionDenied

class SecurityContext:
    """安全上下文管理器"""
    
    def __init__(self, role_mappings: dict = None):
        self.roles = role_mappings or {
            'admin': ['*'],
            'developer': ['api:access', 'data:process'],
            'guest': ['basic:read']
        }

    def has_permission(self, user_roles: list, required: list) -> bool:
        user_perms = set()
        for role in user_roles:
            user_perms.update(self.roles.get(role, []))
        return all(perm in user_perms for perm in required) or '*' in user_perms

def permission_required(permissions: list):
    """权限校验装饰器"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(ctx, *args, **kwargs):
            security = SecurityContext()
            if not security.has_permission(ctx.get('roles', []), permissions):
                raise PermissionDenied(f"缺少权限: {permissions}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
