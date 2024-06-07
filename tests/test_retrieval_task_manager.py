import unittest
import os
import json
from unittest.mock import patch, MagicMock
from .boring import BoringModel
from src.utils.logger import Logger
from src.tasks.retrieval import RetrievalTaskManager

class TestRetrievalTaskManager(unittest.TestCase):

    def setUp(self):
        # Create a mock model
        self.model = BoringModel({'embedding_dim': 768})
        self.model.name = 'boring_model'
        self.logger = Logger(log_dir="./logs")

        # Define the task configuration
        self.task_config = {
            'dataset_path': 'justram/cirr_task7',
            'qrels_config': 'qrels',
            'corpus_config': 'corpus',
            'query_config': 'query',
            'split': 'test',
            'main_metrics': ['map', 'ndcg'],
            'distance_metrics': 'cos',
            'topk': 1000
        }
        
        # Mock load_dataset to avoid actual data loading
        self.load_dataset_patcher = patch('src.tasks.retrieval.load_dataset', side_effect=self.mock_load_dataset)
        self.mock_load_dataset = self.load_dataset_patcher.start()

        # Mock data to be returned by load_dataset
        self.mock_data = {
            'qrels': [{'query_id': '1', 'corpus_id': '10', 'relevance': 1}],
            'corpus': [{'id': '10', 'text': 'some text'}],
            'query': [{'id': '1', 'text': 'some query'}]
        }

        # Define cache folder and callbacks
        self.cache_folder = "./cache"
        self.callbacks = {}
        self.output_folder = "./output"

    def tearDown(self):
        self.load_dataset_patcher.stop()

        # Clean up logs, cache, and output directories
        if os.path.exists(self.logger.log_dir):
            for f in os.listdir(self.logger.log_dir):
                os.remove(os.path.join(self.logger.log_dir, f))
            os.rmdir(self.logger.log_dir)
        
        if os.path.exists(self.cache_folder):
            for root, dirs, files in os.walk(self.cache_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.cache_folder)

        if os.path.exists(self.output_folder):
            for f in os.listdir(self.output_folder):
                os.remove(os.path.join(self.output_folder, f))
            os.rmdir(self.output_folder)

    def mock_load_dataset(self, dataset_path, config, split):
        return self.mock_data[config]

    def test_retrieval_task_manager_run(self):
        # Initialize RetrievalTaskManager
        task_manager = RetrievalTaskManager(
            name='test_task',
            task_config=self.task_config,
            model=self.model,
            logger=self.logger,
            cache_folder=self.cache_folder,
            callbacks=self.callbacks
        )

        # Run the task
        task_manager.run(batch_size=2, output_folder=self.output_folder)

        # Check if the output file is created and contains the expected content
        output_file = os.path.join(self.output_folder, "evaluation_test_task.json")
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            scores = json.load(f)

        # Assuming mock evaluator returns some scores
        self.assertIn('map', scores)
        self.assertIn('ndcg', scores)

if __name__ == '__main__':
    unittest.main()
