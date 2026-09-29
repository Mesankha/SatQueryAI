"""
CROMA-S1 SAR Encoder Loader.
Loads CROMA (ViT-Base/Large) from antofuller/CROMA on Hugging Face.
Pure vision encoder - no language head. Outputs patch tokens for downstream projector.
"""
import os
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from transformers import AutoModel, AutoConfig

from shared.schemas import UniformError, ErrorType
from shared.config import get_config

logger = logging.getLogger(__name__)

# Global singleton state
_croma_model = None
_croma_config = None
_croma_initialized = False
_croma_init_error: Optional[str] = None
_croma_lock = threading.Lock()


def _get_croma_config() -> Dict[str, Any]:
    """Get CROMA configuration from config.yaml."""
    config = get_config()
    return config.get("croma", {})


def load_croma_encoder() -> Optional[UniformError]:
    """Lazy singleton loader for CROMA-S1 encoder.
    
    Loads model only when croma.mock=false. Uses ViT-Base (~350MB) or ViT-Large (~1.2GB).
    Device: CUDA if available, else CPU (slow).
    Returns UniformError if mock mode or load failed, None on success.
    """
    global _croma_model, _croma_config, _croma_initialized, _croma_init_error
    
    with _croma_lock:
        if _croma_initialized:
            return None if _croma_model is not None else UniformError(
                module="M3", error_type=ErrorType.model_unavailable,
                message=f"CROMA init failed: {_croma_init_error}"
            )
        
        croma_config = _get_croma_config()
        if croma_config.get("mock", True):
            logger.info("CROMA mock mode enabled, skipping model load")
            _croma_initialized = True
            return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                               message="CROMA mock mode enabled")
        
        try:
            model_id = croma_config.get("model_id", "antofuller/CROMA")
            variant = croma_config.get("variant", "base")  # "base" or "large"
            device = croma_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            checkpoint_path = croma_config.get("checkpoint_path")
            
            logger.info(f"Loading CROMA-{variant} from {model_id} on {device}")
            
            # Load config first to get architecture details
            _croma_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            
            # Determine model class - CROMA uses ViT backbone
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Load from local checkpoint
                logger.info(f"Loading CROMA weights from local checkpoint: {checkpoint_path}")
                _croma_model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )
                state_dict = torch.load(checkpoint_path, map_location=device)
                _croma_model.load_state_dict(state_dict, strict=False)
            else:
                # Load from HF hub
                _croma_model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )
            
            _croma_model.to(device)
            _croma_model.eval()
            _croma_initialized = True
            logger.info(f"CROMA-{variant} loaded successfully on {device}")
            return None
            
        except Exception as e:
            _croma_init_error = str(e)
            _croma_initialized = True
            logger.error(f"CROMA load failed: {e}")
            return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                               message=f"CROMA load failed: {e}")


def unload_croma() -> None:
    """Free GPU memory and cleanup."""
    global _croma_model, _croma_config, _croma_initialized, _croma_init_error
    
    with _croma_lock:
        if _croma_model is not None:
            _croma_model.cpu()
            del _croma_model
            _croma_model = None
        _croma_config = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _croma_initialized = False
        _croma_init_error = None
        logger.info("CROMA unloaded")


def get_croma_model():
    """Get the loaded CROMA model (loads if needed)."""
    if not _croma_initialized:
        err = load_croma_encoder()
        if err:
            raise RuntimeError(err.message)
    return _croma_model


def get_croma_config():
    """Get the CROMA model config."""
    if not _croma_initialized:
        load_croma_encoder()
    return _croma_config


def run_croma_encoder(image_tensor: torch.Tensor) -> torch.Tensor | UniformError:
    """
    Run CROMA encoder on SAR image tensor.
    
    Args:
        image_tensor: Shape (B, 2, 120, 120) - batch, 2 channels (VV, VH), 120x120
        
    Returns:
        Patch tokens: Shape (B, num_patches, hidden_dim) - e.g., (1, 144, 768) for ViT-B
        or UniformError on failure
    """
    model = get_croma_model()
    if model is None:
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message="CROMA model not loaded")
    
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    try:
        with torch.no_grad():
            # CROMA forward pass - returns patch embeddings
            # The model expects (B, C, H, W) where C=2 for SAR
            outputs = model(image_tensor)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                patch_tokens = outputs[0]  # Usually first element is patch tokens
            elif hasattr(outputs, 'last_hidden_state'):
                patch_tokens = outputs.last_hidden_state
            else:
                patch_tokens = outputs
            
            # Ensure shape is (B, num_patches, hidden_dim)
            if patch_tokens.dim() == 4:
                # If output is (B, H, W, C) or (B, C, H, W), flatten spatial dims
                B, C, H, W = patch_tokens.shape
                patch_tokens = patch_tokens.reshape(B, H * W, C)
            elif patch_tokens.dim() == 3:
                # Already (B, num_patches, hidden_dim)
                pass
            else:
                return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                                   message=f"Unexpected CROMA output shape: {patch_tokens.shape}")
            
            return patch_tokens.cpu()
            
    except Exception as e:
        logger.error(f"CROMA encoder inference failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message=f"CROMA inference failed: {e}")


def get_croma_info() -> Dict[str, Any]:
    """Get info about loaded CROMA model."""
    if not _croma_initialized:
        return {"loaded": False, "error": _croma_init_error}
    
    model = get_croma_model()
    if model is None:
        return {"loaded": False, "error": "Model not loaded"}
    
    return {
        "loaded": True,
        "variant": _get_croma_config().get("variant", "base"),
        "device": str(next(model.parameters()).device),
        "hidden_dim": _croma_config.hidden_size if _croma_config else 768,
        "patch_size": _croma_config.patch_size if _croma_config else 16,
        "num_patches": (120 // (_croma_config.patch_size if _croma_config else 16)) ** 2,
    }