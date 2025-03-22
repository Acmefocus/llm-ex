from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import logging
import redis
import time
from dify.security import get_current_user  # 假设Dify已有的安全模块

# --------------------------
# 基础配置和依赖项
# --------------------------
router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Retrieval"])
logger = logging.getLogger("retrieval")
redis_conn = redis.Redis(host='redis', port=6379, db=0)

# --------------------------
# 数据模型
# --------------------------
class SearchRequest(BaseModel):
    query: str
    knowledge_base_ids: List[str]
    search_type: Optional[str] = "hybrid"  # hybrid/semantic/fulltext
    top_k: Optional[int] = 10

class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict

class FeedbackRequest(BaseModel):
    query: str
    document_id: str
    relevance_score: int  # 1-5

# --------------------------
# 核心检索服务
# --------------------------
class RetrievalService:
    def __init__(self):
        # 初始化各检索模块
        self.cache = CacheManager()
        self.retriever = HybridRetriever()
        self.reranker = ReRanker()
    
    async def search(self, request: SearchRequest) -> List[SearchResult]:
        # 缓存检查
        cache_key = f"search:{request.query}:{':'.join(request.knowledge_base_ids)}"
        if cached := self.cache.get(cache_key):
            logger.info(f"Cache hit for {cache_key}")
            return [SearchResult(**item) for item in cached]
        
        # 执行检索
        start_time = time.time()
        raw_results = await self._execute_search(request)
        processed_results = self.reranker.process(request.query, raw_results)
        
        # 记录日志
        self._log_search(
            request, 
            processed_results,
            duration=time.time()-start_time
        )
        
        # 缓存结果
        self.cache.set(cache_key, [r.dict() for r in processed_results], ttl=300)
        return processed_results
    
    async def _execute_search(self, request: SearchRequest):
        # 根据类型选择检索策略
        if request.search_type == "hybrid":
            return self.retriever.hybrid_search(
                request.query, 
                request.knowledge_base_ids
            )
        elif request.search_type == "semantic":
            return self.retriever.semantic_search(
                request.query,
                request.knowledge_base_ids
            )
        else:
            return self.retriever.fulltext_search(
                request.query,
                request.knowledge_base_ids
            )

# --------------------------
# API端点实现
# --------------------------
@router.post("/search", 
           response_model=List[SearchResult],
           summary="知识库检索接口",
           description="支持混合检索、语义检索和全文检索模式")
async def search_endpoint(
    request: SearchRequest,
    user: dict = Depends(get_current_user)
):
    try:
        service = RetrievalService()
        return await service.search(request)
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="检索服务暂时不可用"
        )

@router.post("/feedback",
           status_code=status.HTTP_204_NO_CONTENT,
           summary="检索结果反馈接口")
async def submit_feedback(
    feedback: FeedbackRequest,
    user: dict = Depends(get_current_user)
):
    try:
        FeedbackService().record_feedback(
            user["id"],
            feedback.query,
            feedback.document_id,
            feedback.relevance_score
        )
    except Exception as e:
        logger.error(f"Feedback submission failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="反馈提交失败"
        )

@router.get("/analytics/queries",
          summary="获取高频查询分析",
          response_model=List[dict])
async def get_query_analytics(
    time_range: str = "7d",
    user: dict = Depends(get_current_user)
):
    try:
        return AnalyticsService().get_top_queries(time_range)
    except Exception as e:
        logger.error(f"Analytics query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="分析数据获取失败"
        )

# --------------------------
# 辅助模块
# --------------------------
class CacheManager:
    """二级缓存管理（内存+Redis）"""
    def __init__(self):
        self.local_cache = {}
    
    def get(self, key):
        # 优先本地缓存
        if result := self.local_cache.get(key):
            return result
        # 其次Redis
        if result := redis_conn.get(key):
            return self._deserialize(result)
        return None
    
    def set(self, key, value, ttl=300):
        # 更新两级缓存
        self.local_cache[key] = value
        redis_conn.setex(key, ttl, self._serialize(value))
    
    def _serialize(self, data):
        return json.dumps(data)
    
    def _deserialize(self, data):
        return json.loads(data)

class FeedbackService:
    """用户反馈处理"""
    def record_feedback(self, user_id, query, doc_id, score):
        # 存储到数据库并更新模型
        self._store_in_db(user_id, query, doc_id, score)
        self._update_reranking_model(query, doc_id, score)
    
    def _update_reranking_model(self, query, doc_id, score):
        # 调用模型微服务更新权重
        pass

# --------------------------
# 文档和测试示例
# --------------------------
"""
API 文档示例：

&zwnj;**搜索接口**&zwnj;：
POST /api/v1/knowledge/search
{
  "query": "如何配置单点登录",
  "knowledge_base_ids": ["kb1", "kb2"],
  "search_type": "hybrid",
  "top_k": 5
}

&zwnj;**响应**&zwnj;：
[
  {
    "id": "doc_123",
    "content": "SSO配置需要...",
    "score": 0.92,
    "metadata": {"source": "kb1", "author": "admin"}
  }
]
"""

# --------------------------
# 测试配置
# --------------------------
import pytest
from fastapi.testclient import TestClient

client = TestClient(router)

def test_search_endpoint():
    response = client.post("/search", json={
        "query": "测试查询",
        "knowledge_base_ids": ["kb1"]
    })
    assert response.status_code == 200
    assert len(response.json()) <= 10

def test_cache_behavior():
    # 首次查询
    response1 = client.post("/search", json={"query": "缓存测试", "knowledge_base_ids": ["kb1"]})
    # 第二次查询应命中缓存
    response2 = client.post("/search", json={"query": "缓存测试", "knowledge_base_ids": ["kb1"]})
    assert response2.headers.get("X-Cache") == "hit"
