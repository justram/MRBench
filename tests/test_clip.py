import unittest
import numpy as np
import torch
from PIL import Image
from src.models.clip import CLIP

class TestCLIP(unittest.TestCase):

    def setUp(self):
        self.model_args = {'model_name': 'openai/clip-vit-base-patch32'}
        self.model = CLIP(self.model_args)
        self.test_image = Image.new('RGB', (224, 224), color='red')

    def test_encode_queries_with_text(self):
        queries = [{'id': 'q1', 'text': 'query 1'}, {'id': 'q2', 'text': 'query 2'}]
        embeddings = self.model.encode_queries(queries, convert_to_tensor=False)
        self.assertEqual(embeddings.shape[1], self.model.get_embedding_dim())
        self.assertIsInstance(embeddings, np.ndarray)

    def test_encode_queries_with_images(self):
        queries = [{'id': 'q1', 'image': self.test_image}, {'id': 'q2', 'image': self.test_image}]
        embeddings = self.model.encode_queries(queries, convert_to_tensor=False)
        self.assertEqual(embeddings.shape[1], self.model.get_embedding_dim())
        self.assertIsInstance(embeddings, np.ndarray)

    def test_encode_queries_with_text_and_images(self):
        queries = [{'id': 'q1', 'text': 'query 1', 'image': self.test_image}, {'id': 'q2', 'text': 'query 2', 'image': self.test_image}]
        embeddings = self.model.encode_queries(queries, convert_to_tensor=False)
        self.assertEqual(embeddings.shape[1], self.model.get_embedding_dim())
        self.assertIsInstance(embeddings, np.ndarray)

    def test_encode_corpus_with_text(self):
        corpus = [{'id': 'd1', 'text': 'document 1'}, {'id': 'd2', 'text': 'document 2'}]
        embeddings = self.model.encode_corpus(corpus, convert_to_tensor=False)
        self.assertEqual(embeddings.shape[1], self.model.get_embedding_dim())
        self.assertIsInstance(embeddings, np.ndarray)

    def test_encode_corpus_with_images(self):
        corpus = [{'id': 'd1', 'image': self.test_image}, {'id': 'd2', 'image': self.test_image}]
        embeddings = self.model.encode_corpus(corpus, convert_to_tensor=False)
        self.assertEqual(embeddings.shape[1], self.model.get_embedding_dim())
        self.assertIsInstance(embeddings, np.ndarray)

    def test_encode_corpus_with_text_and_images(self):
        corpus = [{'id': 'd1', 'text': 'document 1', 'image': self.test_image}, {'id': 'd2', 'text': 'document 2', 'image': self.test_image}]
        embeddings = self.model.encode_corpus(corpus, convert_to_tensor=False)
        self.assertEqual(embeddings.shape[1], self.model.get_embedding_dim())
        self.assertIsInstance(embeddings, np.ndarray)

    def test_get_embedding_dim(self):
        self.assertEqual(self.model.get_embedding_dim(), self.model.embedding_dim)

if __name__ == '__main__':
    unittest.main()
