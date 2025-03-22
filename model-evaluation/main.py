from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate")
async def evaluate_model(config: EvaluationConfig):
    engine = EvaluationEngine()
    result = await engine.evaluate_model_async.delay(config)
    return {"task_id": result.id}

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return result.get()

@app.post("/recommend")
async def recommend_models(requirements: dict):
    recommender = ModelRecommender()
    return recommender.recommend(requirements)
