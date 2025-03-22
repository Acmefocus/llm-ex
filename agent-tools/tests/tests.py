# ---- 单元测试 ----
import pytest

@pytest.mark.asyncio
async def test_http_client():
    executor = ToolExecutor()
    context = {"roles": ["developer"]}
    result = await executor.execute(
        "http_client", 
        {"url": "https://api.example.com/ping"},
        context
    )
    assert "status" in result

def test_permission_check():
    with pytest.raises(PermissionDenied):
        executor.execute(
            "data_processor",
            {"input_data": [], "rules": {}},
            {"roles": ["guest"]}
        )
