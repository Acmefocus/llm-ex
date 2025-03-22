import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

class PatternAnalyzer:
    @staticmethod
    def analyze_frequent_patterns(events_df: pd.DataFrame, min_support=0.05):
        """使用FP-growth挖掘高频操作模式"""
        # 转换为事务列表格式
        transactions = events_df.groupby('session_id')['event_type'].apply(list).tolist()
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # 挖掘频繁项集
        return fpgrowth(df, min_support=min_support, use_colnames=True)
