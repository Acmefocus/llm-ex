from core.decorators import tool

@tool(
    name="data_processor",
    description="数据清洗处理工具",
    permissions=["data_processing"]
)
def process_data(data: list, rules: dict) -> list:
    # 实现数据处理逻辑
    return cleaned_data
