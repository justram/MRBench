import os
import torch
import heapq
import numpy as np
from datasets import Dataset
from collections import defaultdict
from typing import List, Dict, Union, Callable
from src.utils.logger import Logger
from src.evaluator.retrieval.score_fn import score_functions

class ExactSearch:
    '''
    Extend the idea of DenseRetrievalExactSearch from mteb
    '''
    def __init__(
        self,
        model,
        batch_size: int = 128,
        corpus_chunk_size: int = 50000,
        query_chunk_size: int = 1000,
        use_memmap: bool = False,
        memmap_dir: str = "./cache/retrieval",
        save_query_embeddings: bool = False,
        save_corpus_embeddings: bool = False,
        callbacks: Dict[str, Callable] = None,
        logger: Logger = None,
        **kwargs,
    ):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.query_chunk_size = query_chunk_size
        self.use_memmap = use_memmap
        self.memmap_dir = memmap_dir
        self.save_query_embeddings = save_query_embeddings
        self.save_corpus_embeddings = save_corpus_embeddings
        self.callbacks = callbacks or {}
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.corpus_embeddings = defaultdict(list)
        self.results = {}
        self.logger = logger if logger else Logger()
        self.score_functions = score_functions
    
    def search(
        self,
        corpus: Dataset,
        queries: Dataset,
        top_k: int,
        score_function: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if score_function not in self.score_functions:
            raise ValueError(
                f"score function: {score_function} must be either (cos), (dot) or (late_interaction)"
            )

        self.logger.log("Encoding Queries...")
        query_ids = [query["id"] for query in queries]
        self.results = {qid: {} for qid in query_ids}
        queries_list = queries

        query_embeddings = self.encode_and_save(queries_list, query_ids, "queries", self.query_chunk_size, self.save_query_embeddings, **kwargs)

        self.logger.log("Processing Corpus...")
        corpus_ids = [doc["id"] for doc in corpus]
        corpus_list = corpus

        self.logger.log("Encoding Corpus in batches...")
        self.logger.log(f"Scoring Function: {score_function.replace('_', ' ').title()}")

        result_heaps = {qid: [] for qid in query_ids}
        for batch_num, corpus_start_idx in enumerate(range(0, len(corpus_list), self.corpus_chunk_size)):
            self.logger.log(f"Encoding Batch {batch_num + 1}/{len(range(0, len(corpus_list), self.corpus_chunk_size))}...")
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus_list))

            if self.use_memmap and self.check_memmap_exists("corpus", batch_num):
                self.logger.log(f"Loading memmap file for corpus chunk {batch_num}")
                sub_corpus_embeddings = self.get_memmap_embeddings(batch_num, "corpus")
            else:
                self.logger.log(f"Encoding corpus chunk {batch_num}")
                sub_corpus_embeddings = self.encode_and_save(
                    corpus_list.select(range(corpus_start_idx, corpus_end_idx)), 
                    corpus_ids[corpus_start_idx:corpus_end_idx], 
                    "corpus", 
                    self.corpus_chunk_size, 
                    self.save_corpus_embeddings, 
                    chunk_id=batch_num, 
                    **kwargs
                )

            for query_batch_num, query_start_idx in enumerate(range(0, len(query_embeddings), self.query_chunk_size)):
                query_end_idx = min(query_start_idx + self.query_chunk_size, len(query_embeddings))
                sub_query_embeddings = query_embeddings[query_start_idx:query_end_idx]

                scores = self.compute_scores(sub_query_embeddings, sub_corpus_embeddings, score_function)
                scores[torch.isnan(scores)] = -1

                self.update_results(query_ids, query_start_idx, scores, corpus_ids, corpus_start_idx, top_k, result_heaps, return_sorted)

        return self.finalize_results(result_heaps)

    def encode_and_save(self, data_list, data_ids, data_type, chunk_size, save_embeddings, chunk_id=None, **kwargs):
        embeddings = []
        for batch_num, start_idx in enumerate(range(0, len(data_list), chunk_size)):
            end_idx = min(start_idx + chunk_size, len(data_list))
            if data_type == "queries":
                sub_embeddings = self.model.encode_queries(data_list[start_idx:end_idx], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor, **kwargs)
            else:
                sub_embeddings = self.model.encode_corpus(data_list[start_idx:end_idx], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)

            if save_embeddings:
                self.logger.log(f"Saving memmap file for {data_type} chunk {batch_num if chunk_id is None else chunk_id}")
                self.save_memmap_embeddings(sub_embeddings, batch_num if chunk_id is None else chunk_id, data_type)
                embeddings.extend([None] * len(sub_embeddings))  # Placeholder for memmap data
            else:
                embeddings.extend(sub_embeddings)

        if save_embeddings and self.use_memmap:
            self.logger.log(f"Loading memmap embeddings for {data_type}")
            return self.load_memmap_embeddings(data_type)
        
        return np.array(embeddings)

    def compute_scores(self, sub_query_embeddings, sub_corpus_embeddings, score_function):
        if score_function == "late_interaction":
            return self.score_functions["late_interaction"](torch.from_numpy(sub_query_embeddings), torch.from_numpy(sub_corpus_embeddings))
        else:
            return self.score_functions[score_function](torch.from_numpy(sub_query_embeddings), torch.from_numpy(sub_corpus_embeddings))

    def update_results(self, query_ids, query_start_idx, scores, corpus_ids, corpus_start_idx, top_k, result_heaps, return_sorted):
        top_k_values, top_k_idx = torch.topk(scores, min(top_k + 1, len(scores[1]) if len(scores) > 1 else len(scores[-1])), dim=1, largest=True, sorted=return_sorted)
        top_k_values = top_k_values.cpu().tolist()
        top_k_idx = top_k_idx.cpu().tolist()

        for query_itr in range(len(top_k_values)):
            if query_start_idx + query_itr >= len(query_ids):
                continue
            query_id = query_ids[query_start_idx + query_itr]
            for sub_corpus_id, score in zip(top_k_idx[query_itr], top_k_values[query_itr]):
                corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                if len(result_heaps[query_id]) < top_k:
                    heapq.heappush(result_heaps[query_id], (score, corpus_id))
                else:
                    heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

    def finalize_results(self, result_heaps):
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        return self.results

    def load_memmap_embeddings(self, data_type):
        all_embeddings = []
        chunk_id = 0
        while True:
            memmap_file = os.path.join(self.memmap_dir, f"{data_type}_embeddings_{chunk_id}.npy")
            if not os.path.exists(memmap_file):
                self.logger.log(f"Memmap file not found for {data_type} chunk {chunk_id}")
                break
            shape = (self.corpus_chunk_size, self.model.get_embedding_dim()) if data_type == "corpus" else (self.query_chunk_size, self.model.get_embedding_dim())
            file_size = os.path.getsize(memmap_file)
            expected_size = np.prod(shape) * np.dtype('float16').itemsize
            self.logger.log(f"Loading {data_type} chunk {chunk_id}: expected size {expected_size}, file size {file_size}")
            if file_size < expected_size:
                self.logger.log(f"File size mismatch for {data_type} chunk {chunk_id}: expected {expected_size}, got {file_size}")
                raise ValueError(f"File size mismatch for {data_type} chunk {chunk_id}")
            memmap_embeddings = np.memmap(memmap_file, dtype='float16', mode='r', shape=shape)
            all_embeddings.append(memmap_embeddings)
            chunk_id += 1
        if not all_embeddings:
            raise ValueError(f"No memmap files found for {data_type}")
        return np.concatenate(all_embeddings, axis=0)

    def get_memmap_embeddings(self, chunk_id, data_type):
        memmap_file = os.path.join(self.memmap_dir, f"{data_type}_embeddings_{chunk_id}.npy")
        shape = (self.corpus_chunk_size, self.model.get_embedding_dim()) if data_type == "corpus" else (self.query_chunk_size, self.model.get_embedding_dim())
        return np.memmap(memmap_file, dtype='float16', mode='r', shape=shape)

    def save_memmap_embeddings(self, embeddings, chunk_id, data_type):
        memmap_file = os.path.join(self.memmap_dir, f"{data_type}_embeddings_{chunk_id}.npy")
        os.makedirs(self.memmap_dir, exist_ok=True)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy().astype('float16')
        else:
            embeddings = embeddings.astype('float16')
        np.save(memmap_file, embeddings)

    def check_memmap_exists(self, data_type, chunk_id):
        memmap_file = os.path.join(self.memmap_dir, f"{data_type}_embeddings_{chunk_id}.npy")
        return os.path.exists(memmap_file)
