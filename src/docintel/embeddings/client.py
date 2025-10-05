"""BiomedCLIP embedding client for multimodal clinical text and image embeddings.

Direct implementation using open_clip_torch with BiomedCLIP for medical domain
text and image understanding. Falls back to deterministic embeddings for testing.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

try:
    import torch
    from PIL import Image
    import open_clip
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False
    torch = None

from ..config import EmbeddingSettings

_LOGGER = logging.getLogger(__name__)
_FALLBACK_DIMENSION = 512


class EmbeddingClientError(RuntimeError):
    """Raised when the embedding pipeline fails to obtain vectors."""


@dataclass(slots=True)
class EmbeddingResponse:
    """Container for a single embedding vector."""

    embedding: List[float]
    index: int
    model: str
    artefact_type: str = "chunk"
    source_path: Optional[str] = None
    page_reference: Optional[int] = None


class EmbeddingClient:
    """BiomedCLIP multimodal embedding client for clinical text and images."""
    
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
        self._use_fallback = not _HAS_DEPENDENCIES
        
        if _HAS_DEPENDENCIES:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            models_dir = Path("models/biomedclip-cache").resolve()
            if models_dir.exists():
                cache_path = str(models_dir)
                for env_var in ("HUGGINGFACE_HUB_CACHE", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
                    if os.environ.get(env_var) != cache_path:
                        os.environ[env_var] = cache_path
                self._logger.info("embedding | using huggingface cache | path=%s", cache_path)
            
            # Load model if not cached or model name changed
            if (EmbeddingClient._shared_model is None or 
                EmbeddingClient._model_name_cache != settings.embedding_model_name):
                
                try:
                    self._load_biomedclip_model()
                    EmbeddingClient._model_name_cache = settings.embedding_model_name
                except Exception as exc:
                    self._logger.warning(
                        "embedding | BiomedCLIP failed to load, using fallback | error=%s", exc
                    )
                    self._use_fallback = True
        
        if self._use_fallback:
            self._logger.info("embedding | using deterministic fallback embeddings")
            self._embedding_dimension = _FALLBACK_DIMENSION
        else:
            self._embedding_dimension = self._determine_model_dimension()

    def _load_biomedclip_model(self):
        """Load BiomedCLIP model, tokenizer, and preprocessor."""
        self._logger.info(
            "embedding | loading BiomedCLIP | model=%s | device=%s",
            self._settings.embedding_model_name,
            self._device
        )
        
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            self._settings.embedding_model_name
        )
        tokenizer = open_clip.get_tokenizer(self._settings.embedding_model_name)
        
        model.to(self._device)
        model.eval()
        
        EmbeddingClient._shared_model = model
        EmbeddingClient._shared_tokenizer = tokenizer
        EmbeddingClient._shared_preprocess = preprocess_val
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            self._logger.info(
                "embedding | BiomedCLIP loaded | memory_used=%.2fGB", memory_used
            )

    def _determine_model_dimension(self) -> int:
        if not _HAS_DEPENDENCIES or EmbeddingClient._shared_model is None:
            return _FALLBACK_DIMENSION
        model = EmbeddingClient._shared_model
        projection = getattr(model, "text_projection", None)
        if projection is not None and hasattr(projection, "weight"):
            weight = getattr(projection, "weight")
            if hasattr(weight, "shape"):
                return int(weight.shape[1])
        visual = getattr(model, "visual", None)
        if visual is not None:
            output_dim = getattr(visual, "output_dim", None)
            if output_dim is not None:
                try:
                    return int(output_dim)
                except (TypeError, ValueError):
                    pass
        return _FALLBACK_DIMENSION

    async def embed_texts(self, texts: Sequence[str]) -> List[EmbeddingResponse]:
        """Generate embeddings for text sequences."""
        if self._use_fallback:
            return self._generate_fallback_embeddings(texts)
        
        try:
            # Tokenize texts
            text_tokens = EmbeddingClient._shared_tokenizer(
                list(texts), 
                context_length=self._settings.embedding_max_tokens
            ).to(self._device)
            
            # Generate embeddings
            with torch.no_grad():
                text_features = EmbeddingClient._shared_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Convert to response format
            responses = []
            for idx, embedding in enumerate(text_features.cpu().numpy()):
                responses.append(EmbeddingResponse(
                    embedding=embedding.tolist(),
                    index=idx,
                    model=self._settings.embedding_model_name,
                    artefact_type="chunk",
                    source_path=None,
                    page_reference=None,
                ))
            
            return responses
            
        except Exception as exc:
            self._logger.error("embedding | BiomedCLIP batch failed | error=%s", exc)
            return self._generate_fallback_embeddings(texts)

    async def embed_images(self, image_paths: Sequence[Union[str, Path]]) -> List[EmbeddingResponse]:
        """Generate embeddings for image files."""
        if self._use_fallback:
            return self._generate_fallback_embeddings([f"image_{i}" for i in range(len(image_paths))])
        
        try:
            images = []
            valid_indices = []
            
            # Load and preprocess images
            for idx, img_path in enumerate(image_paths):
                try:
                    img_path = Path(img_path)
                    if img_path.exists():
                        image = Image.open(img_path).convert('RGB')
                        processed_img = EmbeddingClient._shared_preprocess(image)
                        images.append(processed_img)
                        valid_indices.append(idx)
                    else:
                        self._logger.warning("embedding | image not found | path=%s", img_path)
                except Exception as exc:
                    self._logger.warning("embedding | failed to load image | path=%s | error=%s", img_path, exc)
            
            if not images:
                return []
            
            # Stack images and move to device
            image_tensor = torch.stack(images).to(self._device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = EmbeddingClient._shared_model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Convert to response format
            responses = []
            for i, (original_idx, embedding) in enumerate(zip(valid_indices, image_features.cpu().numpy())):
                responses.append(EmbeddingResponse(
                    embedding=embedding.tolist(),
                    index=original_idx,
                    model=self._settings.embedding_model_name,
                    artefact_type="figure_image",
                    source_path=str(image_paths[original_idx]),
                    page_reference=None,
                ))
            
            return responses
            
        except Exception as exc:
            self._logger.error("embedding | image embedding failed | error=%s", exc)
            return self._generate_fallback_embeddings([f"image_{i}" for i in range(len(image_paths))])

    def _generate_fallback_embeddings(self, texts: Sequence[str]) -> List[EmbeddingResponse]:
        """Generate deterministic fallback embeddings for testing."""
        responses = []
        for idx, text in enumerate(texts):
            # Create deterministic embedding based on text hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Convert hash to embedding vector
            embedding = []
            for i in range(0, min(len(text_hash), _FALLBACK_DIMENSION * 2), 2):
                hex_pair = text_hash[i:i+2]
                value = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
                embedding.append(value)
            
            # Pad or truncate to exact dimension
            while len(embedding) < _FALLBACK_DIMENSION:
                embedding.append(0.0)
            embedding = embedding[:_FALLBACK_DIMENSION]
            
            # Normalize
            norm = math.sqrt(sum(x*x for x in embedding))
            if norm > 0:
                embedding = [x/norm for x in embedding]
            
            responses.append(EmbeddingResponse(
                embedding=embedding,
                index=idx,
                model="fallback-deterministic",
                artefact_type="chunk",
                source_path=None,
                page_reference=None,
            ))
        
        return responses

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dimension