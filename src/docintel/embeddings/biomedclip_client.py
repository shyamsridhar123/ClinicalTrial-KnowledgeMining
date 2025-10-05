"""BiomedCLIP embedding client for multimodal clinical text and image embeddings."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False

from ..config import EmbeddingSettings

_LOGGER = logging.getLogger(__name__)


class BiomedCLIPClientError(RuntimeError):
    """Raised when the BiomedCLIP embedding pipeline fails."""


@dataclass(slots=True)
class BiomedCLIPResponse:
    """Container for BiomedCLIP embedding responses."""

    embedding: List[float]
    index: int
    model: str
    content_type: str  # 'text' or 'image'


class BiomedCLIPClient:
    """Multimodal embedding client using BiomedCLIP for clinical text and images."""
    
    _shared_model = None
    _shared_tokenizer = None
    _shared_preprocess = None
    _model_name_cache: Optional[str] = None

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._settings = settings
        self._logger = logger or _LOGGER
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not _HAS_OPEN_CLIP:
            raise BiomedCLIPClientError(
                "open_clip_torch is required. Install with: pip install open_clip_torch"
            )
        
        # Set up local model caching
        models_dir = Path("models/biomedclip-cache")
        if models_dir.exists():
            os.environ['HF_HOME'] = str(models_dir.parent)
            os.environ['TRANSFORMERS_CACHE'] = str(models_dir.parent)
        
        # Load model if not cached or model name changed
        if (BiomedCLIPClient._shared_model is None or 
            BiomedCLIPClient._model_name_cache != settings.embedding_model_name):
            
            self._load_model()
            BiomedCLIPClient._model_name_cache = settings.embedding_model_name

    def _load_model(self):
        """Load BiomedCLIP model, tokenizer, and preprocessor."""
        try:
            self._logger.info(
                "biomedclip | loading model | model=%s | device=%s",
                self._settings.embedding_model_name,
                self._device
            )
            
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                self._settings.embedding_model_name
            )
            tokenizer = open_clip.get_tokenizer(self._settings.embedding_model_name)
            
            model.to(self._device)
            model.eval()  # Set to evaluation mode
            
            BiomedCLIPClient._shared_model = model
            BiomedCLIPClient._shared_tokenizer = tokenizer
            BiomedCLIPClient._shared_preprocess = preprocess_val
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                self._logger.info(
                    "biomedclip | model loaded | memory_used=%.2fGB | device=%s",
                    memory_used, self._device
                )
            
        except Exception as exc:
            raise BiomedCLIPClientError(f"Failed to load BiomedCLIP model: {exc}") from exc

    async def embed_texts(self, texts: Sequence[str]) -> List[BiomedCLIPResponse]:
        """Generate embeddings for text sequences."""
        if not texts:
            return []
        
        try:
            # Tokenize texts
            text_tokens = BiomedCLIPClient._shared_tokenizer(
                list(texts), 
                context_length=self._settings.embedding_max_tokens
            ).to(self._device)
            
            # Generate embeddings
            with torch.no_grad():
                text_features = BiomedCLIPClient._shared_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Convert to response format
            responses = []
            for idx, embedding in enumerate(text_features.cpu().numpy()):
                responses.append(BiomedCLIPResponse(
                    embedding=embedding.tolist(),
                    index=idx,
                    model=self._settings.embedding_model_name,
                    content_type="text"
                ))
            
            return responses
            
        except Exception as exc:
            raise BiomedCLIPClientError(f"Text embedding failed: {exc}") from exc

    async def embed_images(self, image_paths: Sequence[Union[str, Path]]) -> List[BiomedCLIPResponse]:
        """Generate embeddings for image files."""
        if not image_paths:
            return []
        
        try:
            images = []
            valid_indices = []
            
            # Load and preprocess images
            for idx, img_path in enumerate(image_paths):
                try:
                    img_path = Path(img_path)
                    if img_path.exists():
                        image = Image.open(img_path).convert('RGB')
                        processed_img = BiomedCLIPClient._shared_preprocess(image)
                        images.append(processed_img)
                        valid_indices.append(idx)
                    else:
                        self._logger.warning("biomedclip | image not found | path=%s", img_path)
                except Exception as exc:
                    self._logger.warning("biomedclip | failed to load image | path=%s | error=%s", img_path, exc)
            
            if not images:
                return []
            
            # Stack images and move to device
            image_tensor = torch.stack(images).to(self._device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = BiomedCLIPClient._shared_model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Convert to response format
            responses = []
            for i, (original_idx, embedding) in enumerate(zip(valid_indices, image_features.cpu().numpy())):
                responses.append(BiomedCLIPResponse(
                    embedding=embedding.tolist(),
                    index=original_idx,
                    model=self._settings.embedding_model_name,
                    content_type="image"
                ))
            
            return responses
            
        except Exception as exc:
            raise BiomedCLIPClientError(f"Image embedding failed: {exc}") from exc

    async def compute_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between text and image embeddings."""
        return torch.cosine_similarity(
            text_embeddings.unsqueeze(1), 
            image_embeddings.unsqueeze(0), 
            dim=2
        )

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension for BiomedCLIP."""
        return 512  # BiomedCLIP standard dimension