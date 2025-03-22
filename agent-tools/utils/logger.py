# ---- 日志管理 ----
# utils/logger.py
import logging

class ToolLogger:
    def __init__(self):
        self.logger = logging.getLogger("AgentTools")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_invocation(self, tool_name: str, params: dict):
        self.logger.info(f"Invoking {tool_name} with params {params}")
    
    def log_success(self, tool_name: str):
        self.logger.info(f"Tool {tool_name} executed successfully")
    
    def log_failure(self, tool_name: str, error: str):
        self.logger.error(f"Tool {tool_name} failed: {error}")
