
# ---- 示例工具实现 ----
# plugins/api_tools.py
from core.decorators import tool

@tool(
    name="http_client",
    description="HTTP客户端工具",
    permissions=["api:access"],
    cache={"ttl": 60}
)
async def http_client(url: str, method: str = "GET", headers: dict = None) -> dict:
    import requests
    response = requests.request(method, url, headers=headers)
    response.raise_for_status()
    return response.json()

@tool(
    name="data_processor",
    description="数据清洗工具",
    permissions=["data:process"]
)
def data_processor(input_data: list, rules: dict) -> list:
    return [item for item in input_data if item[rules['field']] == rules['value']]
