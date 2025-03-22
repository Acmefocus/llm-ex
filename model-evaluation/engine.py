from concurrent.futures import ThreadPoolExecutor
from celery import Celery

celery_app = Celery('evaluation_worker', broker='redis://localhost:6379/0')

class EvaluationEngine:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    @celery_app.task
    def evaluate_model_async(self, model, test_set, metrics):
        return self._evaluate(model, test_set, metrics)
    
    def batch_evaluate(self, models, test_set, metrics):
        futures = []
        for model in models:
            future = self.executor.submit(
                self._evaluate, model, test_set, metrics
            )
            futures.append(future)
        return [f.result() for f in futures]
    
    def _evaluate(self, model, test_set, metrics):
        evaluators = self._get_evaluators(metrics, test_set)
        results = {}
        for metric, evaluator in evaluators.items():
            results[metric] = evaluator.evaluate(model)
        return EvaluationResult(
            model=model.name,
            scores=results,
            metadata={"test_set": test_set.id},
            timestamp=datetime.now().isoformat()
        )
