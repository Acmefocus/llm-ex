import os
from datetime import timedelta

class Config:
    # Redis配置
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = 6379
    SESSION_EXPIRE = timedelta(minutes=30)  # 会话过期时间

    # 数据匿名化盐值
    ANONYMIZE_SALT = "dify_analytics_salt_2023"

    # 机器学习模型路径
    MODEL_PATH = "models/"
