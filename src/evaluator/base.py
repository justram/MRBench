from abc import ABC, abstractmethod
from typing import List, Dict

class BaseEvaluator(ABC):

    @abstractmethod
    def parse_results(self, results: List[str]) -> Dict[str, Dict[str, float]]:
        pass

    @abstractmethod
    def evaluate(self, results: List[str], metrics: List[str]) -> Dict[str, float]:
        pass