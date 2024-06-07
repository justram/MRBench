import ranx
from typing import List, Dict
from src.evaluator.base import BaseEvaluator

class RetrievalEvaluator(BaseEvaluator):
    def __init__(self, qrels: Dict, metrics: List[str]):
        super().__init__()
        self.qrels = ranx.Qrels(qrels)
        self.metrics = metrics

    def parse_results(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        parsed_results = {}
        for result in results:
            query_id, corpus_id, rank, score, tag = result.split(', ')
            if query_id not in parsed_results:
                parsed_results[query_id] = {}
            parsed_results[query_id][corpus_id] = float(score)
        return parsed_results

    def evaluate(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        run = ranx.Run(results)
        scores = ranx.evaluate(self.qrels, run, self.metrics)
        return scores
