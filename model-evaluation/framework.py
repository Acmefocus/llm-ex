from typing import Dict, List, Callable
from pydantic import BaseModel

class EvaluationConfig(BaseModel):
    model_name: str
    test_set: str
    metrics: List[str]
    priority_weights: Dict[str, float] = None

class EvaluationResult(BaseModel):
    model: str
    scores: Dict[str, float]
    metadata: dict
    timestamp: str

class BaseEvaluator:
    def __init__(self, test_cases: List[dict]):
        self.test_cases = test_cases
    
    def evaluate(self, model: Callable) -> Dict[str, float]:
        raise NotImplementedError

class AccuracyEvaluator(BaseEvaluator):
    def evaluate(self, model):
        correct = 0
        for case in self.test_cases:
            response = model.generate(case["input"])
            if response.strip() == case["expected"].strip():
                correct += 1
        return {"accuracy": correct / len(self.test_cases)}

class CreativityEvaluator(BaseEvaluator):
    def evaluate(self, model):
        # 使用LLM评估创造性的实现
        pass
