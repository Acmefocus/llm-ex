from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ModelRecommender:
    def __init__(self, evaluation_history):
        self.history = evaluation_history
    
    def recommend(self, requirements: dict, top_n=3):
        model_vectors = self._create_feature_vectors()
        req_vector = self._create_requirement_vector(requirements)
        
        similarities = cosine_similarity([req_vector], model_vectors)
        sorted_indices = np.argsort(similarities)[::-1][:top_n]
        
        return [self.models[i] for i in sorted_indices]
    
    def _create_feature_vectors(self):
        # 将评估结果转换为特征向量
        pass
    
    def _create_requirement_vector(self, req):
        # 将用户需求转换为特征向量
        pass
