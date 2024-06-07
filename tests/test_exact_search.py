import unittest
import numpy as np
import os
import psutil
from unittest.mock import MagicMock
from src.evaluator.retrieval.exact_search import ExactSearch
from boring import BoringModel
from src.utils.logger import Logger

class TestExactSearch(unittest.TestCase):

    def setUp(self):
        self.model = BoringModel({'embedding_dim': 768})
        self.model.encode_queries = MagicMock(return_value=np.random.rand(10, 768))
        self.model.encode_corpus = MagicMock(return_value=np.random.rand(50, 768))
        self.logger = Logger(log_dir="./logs")
        self.memmap_dir = "./cache/retrieval/test"

    def tearDown(self):
        if os.path.exists(self.memmap_dir):
            for f in os.listdir(self.memmap_dir):
                os.remove(os.path.join(self.memmap_dir, f))
            os.rmdir(self.memmap_dir)

    def test_search(self):
        searcher = ExactSearch(
            model=self.model,
            batch_size=10,
            use_memmap=True,
            memmap_dir=self.memmap_dir,
            logger=self.logger
        )
        corpus = [{'id': f'doc{i}', 'text': f'text {i}'} for i in range(50)]
        queries = [{'id': f'q{i}', 'text': f'query {i}'} for i in range(10)]
        results = searcher.search(
            corpus=corpus,
            queries=queries,
            top_k=5,
            score_function="cos_sim"
        )
        self.assertEqual(len(results), 10)
        self.assertTrue(all(len(res) <= 5 for res in results.values()))

    def test_memmap_save_and_load(self):
        searcher = ExactSearch(
            model=self.model,
            batch_size=10,
            corpus_chunk_size=50,  # Ensure this is consistent with corpus size
            query_chunk_size=10,   # Ensure this is consistent with query size
            use_memmap=True,
            memmap_dir=self.memmap_dir,
            logger=self.logger
        )
        corpus = [{'id': f'doc{i}', 'text': f'text {i}'} for i in range(50)]
        queries = [{'id': f'q{i}', 'text': f'query {i}'} for i in range(10)]
        
        # Save embeddings
        searcher.encode_and_save(corpus, [doc['id'] for doc in corpus], "corpus", 50, save_embeddings=True, chunk_id=0)
        searcher.encode_and_save(queries, [query['id'] for query in queries], "queries", 10, save_embeddings=True, chunk_id=0)

        # Load embeddings
        corpus_embeddings = searcher.get_memmap_embeddings(0, "corpus")
        query_embeddings = searcher.get_memmap_embeddings(0, "queries")

        self.assertEqual(corpus_embeddings.shape, (50, self.model.get_embedding_dim()))
        self.assertEqual(query_embeddings.shape, (10, self.model.get_embedding_dim()))

    def test_memory_usage(self):
        searcher = ExactSearch(
            model=self.model,
            batch_size=10,
            use_memmap=True,
            memmap_dir=self.memmap_dir,
            logger=self.logger
        )
        corpus = [{'id': f'doc{i}', 'text': f'text {i}'} for i in range(10000)]
        queries = [{'id': f'q{i}', 'text': f'query {i}'} for i in range(100)]

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        print(f"Memory usage before search: {memory_before / (1024 * 1024)} MB")

        results = searcher.search(
            corpus=corpus,
            queries=queries,
            top_k=5,
            score_function="cos_sim"
        )

        memory_after = process.memory_info().rss
        print(f"Memory usage after search: {memory_after / (1024 * 1024)} MB")

        self.assertLess(memory_after, memory_before * 1.5, "Memory usage increased too much after search")

if __name__ == '__main__':
    unittest.main()
