import redis
import hashlib
import uuid
import time
from config import Config

class SessionTracker:
    def __init__(self):
        self.redis = redis.StrictRedis(
            host=Config.REDIS_HOST, 
            port=Config.REDIS_PORT, 
            decode_responses=True
        )

    def _generate_session_id(self, user_id: str) -> str:
        """生成匿名化会话ID"""
        raw_id = f"{user_id}-{uuid.uuid4()}-{time.time()}"
        return hashlib.sha256(
            (raw_id + Config.ANONYMIZE_SALT).encode()
        ).hexdigest()

    def track_event(self, user_id: str, event_type: str, metadata: dict):
        """记录用户事件"""
        session_id = self._generate_session_id(user_id)
        event_key = f"session:{session_id}:events"
        
        # 存储事件数据（带时间戳）
        event_data = {
            "timestamp": time.time(),
            "event_type": event_type,
            "metadata": str(metadata)
        }
        self.redis.rpush(event_key, str(event_data))
        
        # 更新会话过期时间
        self.redis.expire(event_key, Config.SESSION_EXPIRE.total_seconds())
