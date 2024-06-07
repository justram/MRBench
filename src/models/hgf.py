import torch
from transformers import AutoModel, AutoProcessor
from typing import List, Dict, Union
from PIL import Image
from src.models.base import BaseModel
from tqdm.auto import tqdm

class HgfModel(BaseModel):
    '''Models compatible with CLIP
    '''
    def __init__(self, model_args: Dict):
        super().__init__(model_args)
        self.name = model_args['model_name_or_path'].split('/')[-1]
        self.model = AutoModel.from_pretrained(model_args['model_name_or_path'])
        self.processor = AutoProcessor.from_pretrained(model_args['model_name_or_path'])
        self.embedding_dim = self.model.config.projection_dim if hasattr(self.model.config, 'projection_dim') else self.model.config.text_config.hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.prompt = 'A picture of'

    @torch.inference_mode()
    def encode(self, data, batch_size: int = 32, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        embeddings = []
        progress_bar = tqdm(range(0, len(data['id']), batch_size), desc='Encoding...', disable=not show_progress_bar)
        
        for start_idx in progress_bar:
            end_idx = start_idx + batch_size
            batch = {k: v[start_idx:end_idx] for k, v in data.items()}

            texts = batch.get('text', None)
            images = batch.get('image', None)
            has_images = any(img is not None for img in images)

            if any(texts):
                texts = [f'{self.prompt} {t}' for t in texts]
                text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            else:
                text_inputs = None

            image_inputs = self.processor(images=[img for img in images if img is not None], return_tensors="pt").to(self.device) if has_images else None

            text_embeddings = self.model.get_text_features(**text_inputs) if text_inputs else None
            image_embeddings = self.model.get_image_features(**image_inputs) if has_images else None

            if text_embeddings is not None and image_embeddings is not None:
                batch_embeddings = (text_embeddings + image_embeddings) / 2
            elif text_embeddings is not None:
                batch_embeddings = text_embeddings
            else:
                batch_embeddings = image_embeddings

            embeddings.append(batch_embeddings)

        embeddings = torch.cat(embeddings, dim=0).detach().cpu()
        return embeddings if convert_to_tensor else embeddings.numpy()

    def encode_queries(self, queries, batch_size: int = 32, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.encode(queries, batch_size, show_progress_bar, convert_to_tensor, **kwargs)

    def encode_corpus(self, corpus, batch_size: int = 32, show_progress_bar: bool = True, convert_to_tensor: bool = True, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.encode(corpus, batch_size, show_progress_bar, convert_to_tensor, **kwargs)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
