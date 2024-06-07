import numpy as np
import torch
from typing import List, Dict, Union

class BaseModel:
    def __init__(self, model_args: Dict):
        self.model_args = model_args
        self.name = model_args.get('model_name_or_path', 'base_model')

    def encode_queries(self, queries: List[Dict[str, str]], batch_size: int = 32, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        """
        Encodes a list of queries into embeddings.

        Args:
            queries (List[Dict[str, str]]): List of queries to encode.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show a progress bar.
            convert_to_tensor (bool): Whether to convert embeddings to torch.Tensor.
            **kwargs: Additional arguments.

        Returns:
            Union[np.ndarray, torch.Tensor]: Encoded embeddings.
        """
        raise NotImplementedError("The method encode_queries() is not implemented.")

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 32, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        """
        Encodes a list of corpus documents into embeddings.

        Args:
            corpus (List[Dict[str, str]]): List of corpus documents to encode.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show a progress bar.
            convert_to_tensor (bool): Whether to convert embeddings to torch.Tensor.
            **kwargs: Additional arguments.

        Returns:
            Union[np.ndarray, torch.Tensor]: Encoded embeddings.
        """
        raise NotImplementedError("The method encode_corpus() is not implemented.")

    def get_embedding_dim(self) -> int:
        """
        Returns the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings.
        """
        raise NotImplementedError("The method get_embedding_dim() is not implemented.")
