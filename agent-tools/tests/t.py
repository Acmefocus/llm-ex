
# ---- 使用示例 ----
from core.executor import ToolExecutor
from core.security import SecurityContext

executor = ToolExecutor()

# 创建用户上下文
user_context = {
    "user_id": "user_123",
    "roles": ["developer"]
}

# 执行工具
try:
    result = await executor.execute(
        tool_name="http_client",
        params={"url": "https://api.example.com/data", "method": "GET"},
        context=user_context
    )
    print(result)
except Exception as e:
    print(f"执行失败: {str(e)}")

