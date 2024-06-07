import unittest
import numpy as np
import torch
from src.evaluator.retrieval.score_fn import cos_sim, dot_score, late_interaction_score

class TestScoreFunctions(unittest.TestCase):

    def setUp(self):
        self.query_embeddings = torch.tensor(np.random.rand(10, 768), dtype=torch.float)
        self.corpus_embeddings = torch.tensor(np.random.rand(50, 768), dtype=torch.float)

    def test_cos_sim(self):
        scores = cos_sim(self.query_embeddings, self.corpus_embeddings)
        self.assertEqual(tuple(scores.size()), (10, 50))
        self.assertIsInstance(scores, torch.Tensor)

    def test_dot_score(self):
        scores = dot_score(self.query_embeddings, self.corpus_embeddings)
        self.assertEqual(tuple(scores.size()), (10, 50))
        self.assertIsInstance(scores, torch.Tensor)

    def test_late_interaction_score(self):
        q_reps = torch.tensor(np.random.rand(10, 5, 768), dtype=torch.float)
        p_reps = torch.tensor(np.random.rand(50, 5, 768), dtype=torch.float)
        scores = late_interaction_score(q_reps, p_reps)
        self.assertEqual(tuple(scores.size()), (10, 50))
        self.assertIsInstance(scores, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
