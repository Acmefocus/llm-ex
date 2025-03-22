# ---- 部署配置 ----
# config.py
import os

class Config:
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() == "true"
