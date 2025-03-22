

# ---- 启动入口 ----
# main.py
import uvicorn
from fastapi import FastAPI
from core.executor import ToolExecutor

app = FastAPI()
executor = ToolExecutor()

@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, params: dict, context: dict):
    return await executor.execute(tool_name, params, context)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
