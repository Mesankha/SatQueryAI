"""
M2 Retrieval - DOFA Embedder.
DOFA (Domain-adaptive Foundation model for Agriculture) for satellite image embeddings.
Used for image↔image similarity only. Does not provide text-image retrieval.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global model singleton
_dofa_model: Optional[torch.nn.Module] = None
_dofa_device: Optional[torch.device] = None
_transform = None


def _get_transform():
    """Get or create the image transform for DOFA."""
    global _transform
    if _transform is None:
        from torchvision import transforms
        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224), antialias=True),
        ])
    return _transform


def load_dofa() -> None:
    """
    Lazy singleton loader for DOFA model.
    Uses timm vit_base_patch16_224 with tum-dofa/dofa-vit-base weights.
    """
    global _dofa_model, _dofa_device

    if _dofa_model is not None:
        return

    try:
        import timm
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        logger.warning(f"DOFA dependencies not available: {e}")
        return

    # Determine device
    _dofa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading DOFA on {_dofa_device}")

    try:
        # Create base ViT model
        _dofa_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)

        # Load DOFA weights from Hugging Face Hub
        weights_path = hf_hub_download(
            repo_id="tum-dofa/dofa-vit-base",
            filename="pytorch_model.bin",
        )
        state_dict = torch.load(weights_path, map_location=_dofa_device, weights_only=True)

        # Handle potential key mismatches
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = _dofa_model.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"DOFA missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"DOFA unexpected keys: {unexpected[:5]}...")

        _dofa_model.to(_dofa_device)
        _dofa_model.eval()
        logger.info("DOFA model loaded successfully")

    except Exception as e:
        logger.warning(f"Failed to load DOFA model: {e}")
        _dofa_model = None
        _dofa_device = None


def unload_dofa() -> None:
    """Free GPU memory and clear model."""
    global _dofa_model, _dofa_device
    if _dofa_model is not None:
        _dofa_model.cpu()
        del _dofa_model
        _dofa_model = None
    if _dofa_device is not None and _dofa_device.type == "cuda":
        torch.cuda.empty_cache()
    _dofa_device = None
    logger.info("DOFA model unloaded")


def embed_images(
    paths: list[str],
    wavelengths_nm: list[list[float]],
    batch_size: int = 16,
) -> np.ndarray:
    """
    Embed a list of image paths using DOFA.

    Args:
        paths: List of image file paths (GeoTIFF, PNG, JPEG)
        wavelengths_nm: List of wavelength lists per image (for multi-spectral)
                       Not used for DOFA (RGB-only) but kept for API compatibility
        batch_size: Batch size for inference

    Returns:
        np.ndarray of shape (N, D) float32, L2-normalized embeddings
        Unreadable files produce zero vectors and are logged.
    """
    load_dofa()

    if _dofa_model is None:
        logger.warning("DOFA model not available, returning zero embeddings")
        return np.zeros((len(paths), 768), dtype=np.float32)

    transform = _get_transform()
    embeddings = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                # Read image using rasterio for GeoTIFF or PIL for others
                if path.lower().endswith((".tif", ".tiff")):
                    import rasterio
                    with rasterio.open(path) as src:
                        # Read first 3 bands as RGB
                        data = src.read([1, 2, 3])
                        # Normalize to 0-1 range
                        data = data.astype(np.float32)
                        for b in range(3):
                            bmin, bmax = data[b].min(), data[b].max()
                            if bmax > bmin:
                                data[b] = (data[b] - bmin) / (bmax - bmin)
                        # Convert to PIL-like format (H, W, C)
                        img = np.transpose(data, (1, 2, 0))
                        # Convert to tensor
                        from PIL import Image
                        img = Image.fromarray((img * 255).astype(np.uint8))
                else:
                    from PIL import Image
                    img = Image.open(path).convert("RGB")

                tensor = transform(img)
                batch_tensors.append(tensor)

            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                # Add zero tensor as placeholder
                batch_tensors.append(torch.zeros(3, 224, 224))

        if not batch_tensors:
            embeddings.append(np.zeros((len(batch_paths), 768), dtype=np.float32))
            continue

        batch = torch.stack(batch_tensors).to(_dofa_device)

        with torch.no_grad():
            batch_emb = _dofa_model(batch)
            # L2 normalize
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            embeddings.append(batch_emb.cpu().numpy())

    if embeddings:
        return np.vstack(embeddings).astype(np.float32)
    else:
        return np.zeros((len(paths), 768), dtype=np.float32)