import os
import json
from pathlib import Path
from datasets import load_dataset
from typing import Dict, Callable, List
from .task_manager import TaskManager
from src.evaluator.retrieval.exact_search import ExactSearch
from src.evaluator.retrieval.evaluator import RetrievalEvaluator

class RetrievalTaskManager(TaskManager):
    def __init__(self, name: str, task_config: Dict, model, logger, cache_folder: str, callbacks: Dict[str, Callable]):
        super().__init__(name, task_config, model, logger)
        self.dataset_type = task_config['type']
        self.dataset_path = task_config['dataset_path']
        self.qrels_config = task_config['qrels_config']
        self.corpus_config = task_config['corpus_config']
        self.query_config = task_config['query_config']
        self.split = task_config['split']
        
        self.main_metrics = task_config['main_metrics']
        self.distance_metrics = task_config['distance_metrics']
        self.topk = task_config['topk']
        self.cache_folder = cache_folder
        self.callbacks = callbacks
        self.qrels = self.load_qrels()
        self.corpus = self.load_corpus()
        self.queries = self.load_queries()

    def load_qrels(self) -> Dict:
        if self.dataset_type == 'hgf':
            dataset = load_dataset(self.dataset_path, name=self.qrels_config, split=self.split)
        if self.dataset_type == 'parquet':
            data_files = [str(p) for p in Path(self.dataset_path).glob(f'{self.qrels_config}*.parquet')]
            dataset = load_dataset('parquet', data_files=sorted(data_files), split='train')
        return {str(row['query-id']): {str(row['corpus-id']): row['score']} for row in dataset}

    def load_corpus(self) -> List[Dict[str, str]]:
        if self.dataset_type == 'hgf':
            dataset = load_dataset(self.dataset_path, name=self.corpus_config, split='corpus')
        if self.dataset_type == 'parquet':
            data_files = [str(p) for p in Path(self.dataset_path).glob(f'{self.corpus_config}*.parquet')]
            dataset = load_dataset('parquet', data_files=sorted(data_files), split='train')
        return dataset

    def load_queries(self) -> List[Dict[str, str]]:
        if self.dataset_type == 'hgf':
            dataset = load_dataset(self.dataset_path, name=self.query_config, split=self.split)
        if self.dataset_type == 'parquet':
            data_files = [str(p) for p in Path(self.dataset_path).glob(f'{self.query_config}*.parquet')]
            dataset = load_dataset('parquet', data_files=sorted(data_files), split='train')
        return dataset

    def run(self, batch_size: int, output_folder: str):
        
        searcher = ExactSearch(
            model=self.model,
            batch_size=batch_size,
            use_memmap=True,
            memmap_dir=os.path.join(self.cache_folder, f"retrieval/{self.name}/{self.model.name}"),
            logger=self.logger,
            callbacks=self.callbacks
        )
        
        results = searcher.search(
            corpus=self.corpus,
            queries=self.queries,
            top_k=self.topk,
            score_function=self.distance_metrics,
            return_sorted=True
        )

        self.logger.log('Running evaluation...')
        evaluator = RetrievalEvaluator(self.qrels, self.main_metrics)
        scores = evaluator.evaluate(results)

        os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists
        output_file = os.path.join(output_folder, f"evaluation_{self.name}.json")
        with open(output_file, "w") as f:
            json.dump(scores, f, indent=4)
            
        self.logger.log(scores)
        self.logger.log(f"Evaluation for task '{self.name}' saved to {output_file}")

