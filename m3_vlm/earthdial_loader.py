"""
EarthDial VLM Loader - Replaces GeoChat.
Loads EarthDial (InternViT + Qwen) from HF.
Exposes optical encoder, shared decoder for fusion with CROMA SAR tokens.
"""
import os
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import AutoModel, AutoProcessor, AutoConfig, AutoTokenizer

from shared.schemas import UniformError, ErrorType
from shared.config import get_config

logger = logging.getLogger(__name__)

# Global singleton state
_earthdial_model = None
_earthdial_processor = None
_earthdial_initialized = False
_earthdial_init_error: Optional[str] = None
_earthdial_lock = threading.Lock()


def _get_earthdial_config() -> Dict[str, Any]:
    """Get EarthDial configuration from config.yaml."""
    config = get_config()
    return config.get("earthdial", {})


def load_earthdial() -> Optional[UniformError]:
    """Lazy singleton loader for EarthDial model.
    
    Loads model only when earthdial.mock=false. Uses 4-bit quantization.
    Device: CUDA if available, else CPU (slow).
    Returns UniformError if mock mode or load failed, None on success.
    """
    global _earthdial_model, _earthdial_processor, _earthdial_initialized, _earthdial_init_error
    
    with _earthdial_lock:
        if _earthdial_initialized:
            return None if _earthdial_model is not None else UniformError(
                module="M3", error_type=ErrorType.model_unavailable,
                message=f"EarthDial init failed: {_earthdial_init_error}"
            )
        
        ed_config = _get_earthdial_config()
        if ed_config.get("mock", True):
            logger.info("EarthDial mock mode enabled, skipping model load")
            _earthdial_initialized = True
            return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                               message="EarthDial mock mode enabled")
        
        try:
            model_id = ed_config.get("model_id", "hiyamdebary/EarthDial")
            checkpoint_name = ed_config.get("checkpoint", "EarthDial_4B_RGB")
            load_in_4bit = ed_config.get("load_in_4bit", True)
            device = ed_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            lora_adapter_path = ed_config.get("lora_adapter_path")
            
            logger.info(f"Loading EarthDial {checkpoint_name} from {model_id} on {device} (4-bit={load_in_4bit})")
            
            # Load processor (tokenizer + image processor)
            _earthdial_processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            
            # Set pad token
            if _earthdial_processor.tokenizer.pad_token is None:
                tok = _earthdial_processor.tokenizer
                if tok.unk_token is not None:
                    tok.pad_token = tok.unk_token
                else:
                    tok.add_special_tokens({"pad_token": "<pad>"})
            
            # Load model with quantization
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                
                _earthdial_model = AutoModel.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map={"": 0} if device == "cuda" else "cpu",
                    trust_remote_code=True,
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                _earthdial_model = AutoModel.from_pretrained(
                    model_id,
                    device_map={"": 0} if device == "cuda" else "cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                )
            
            # Resize embeddings if pad token was added
            if _earthdial_processor.tokenizer.pad_token == "<pad>":
                _earthdial_model.resize_token_embeddings(len(_earthdial_processor.tokenizer))
            
            # Load LoRA adapter if present
            if lora_adapter_path and os.path.exists(lora_adapter_path):
                logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
                from peft import PeftModel
                _earthdial_model = PeftModel.from_pretrained(_earthdial_model, lora_adapter_path, is_trainable=False)
            
            _earthdial_model.eval()
            _earthdial_initialized = True
            logger.info("EarthDial loaded successfully")
            return None
            
        except Exception as e:
            _earthdial_init_error = str(e)
            _earthdial_initialized = True
            logger.error(f"EarthDial load failed: {e}")
            return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                               message=f"EarthDial load failed: {e}")


def unload_earthdial() -> None:
    """Free GPU memory and cleanup."""
    global _earthdial_model, _earthdial_processor, _earthdial_initialized, _earthdial_init_error
    
    with _earthdial_lock:
        if _earthdial_model is not None:
            _earthdial_model.cpu()
            del _earthdial_model
            _earthdial_model = None
        if _earthdial_processor is not None:
            del _earthdial_processor
            _earthdial_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _earthdial_initialized = False
        _earthdial_init_error = None
        logger.info("EarthDial unloaded")


def get_earthdial_model():
    """Get the loaded EarthDial model (loads if needed)."""
    if not _earthdial_initialized:
        err = load_earthdial()
        if err:
            raise RuntimeError(err.message)
    return _earthdial_model


def get_earthdial_processor():
    """Get the EarthDial processor."""
    if not _earthdial_initialized:
        load_earthdial()
    return _earthdial_processor


def get_earthdial_info() -> Dict[str, Any]:
    """Get info about loaded EarthDial model."""
    if not _earthdial_initialized:
        return {"loaded": False, "error": _earthdial_init_error}
    
    model = get_earthdial_model()
    if model is None:
        return {"loaded": False, "error": "Model not loaded"}
    
    return {
        "loaded": True,
        "checkpoint": _get_earthdial_config().get("checkpoint", "EarthDial_4B_RGB"),
        "device": str(next(model.parameters()).device),
        "model_type": type(model).__name__,
    }


# =============================================================================
# OPTICAL ENCODER (InternViT) - for optical images
# =============================================================================

def encode_optical_image(image_tensor: torch.Tensor) -> torch.Tensor | UniformError:
    """
    Encode optical image through EarthDial's InternViT encoder.
    
    Args:
        image_tensor: Shape (B, 3, H, W) - RGB optical image, typically 448x448 or 224x224
        
    Returns:
        Patch tokens: Shape (B, num_patches, hidden_dim) - e.g., (1, 256, 1024) for InternViT-6B
        or UniformError on failure
    """
    model = get_earthdial_model()
    if model is None:
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message="EarthDial model not loaded")
    
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    try:
        with torch.no_grad():
            # EarthDial uses InternViT as vision encoder
            # Access the vision tower/encoder
            if hasattr(model, 'vision_tower'):
                vision_encoder = model.vision_tower
            elif hasattr(model, 'vision_encoder'):
                vision_encoder = model.vision_encoder
            elif hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
                vision_encoder = model.model.vision_tower
            else:
                # Try to find vision encoder in model structure
                vision_encoder = _find_vision_encoder(model)
            
            if vision_encoder is None:
                return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                                   message="Could not find vision encoder in EarthDial model")
            
            # Forward through vision encoder
            outputs = vision_encoder(pixel_values=image_tensor)
            
            # Extract patch tokens (remove CLS token if present)
            if hasattr(outputs, 'last_hidden_state'):
                patch_tokens = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                patch_tokens = outputs[0]
            else:
                patch_tokens = outputs
            
            # Remove CLS token if present (first token)
            if patch_tokens.shape[1] > 1:
                # Check if first token is CLS by comparing norms
                cls_token = patch_tokens[:, 0:1, :]
                patch_tokens = patch_tokens[:, 1:, :]  # Remove CLS
            
            return patch_tokens.cpu()
            
    except Exception as e:
        logger.error(f"EarthDial optical encoding failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message=f"EarthDial optical encoding failed: {e}")


def _find_vision_encoder(model) -> Optional[torch.nn.Module]:
    """Find vision encoder in model structure."""
    # Common attribute names for vision encoder
    for attr in ['vision_tower', 'vision_encoder', 'visual', 'vit', 'encoder']:
        if hasattr(model, attr):
            return getattr(model, attr)
    
    # Check nested
    if hasattr(model, 'model'):
        for attr in ['vision_tower', 'vision_encoder', 'visual', 'vit', 'encoder']:
            if hasattr(model.model, attr):
                return getattr(model.model, attr)
    
    return None


# =============================================================================
# SHARED DECODER (Qwen) - for text generation from fused tokens
# =============================================================================

def get_shared_decoder() -> Tuple[Optional[torch.nn.Module], Optional[Any]]:
    """
    Get the shared language decoder (Qwen) and tokenizer from EarthDial.
    
    Returns:
        (decoder_module, tokenizer) or (None, None) if not loaded
    """
    model = get_earthdial_model()
    processor = get_earthdial_processor()
    
    if model is None or processor is None:
        return None, None
    
    # Find the language model / decoder
    decoder = None
    for attr in ['language_model', 'llm', 'decoder', 'model']:
        if hasattr(model, attr):
            decoder = getattr(model, attr)
            break
    
    if decoder is None and hasattr(model, 'model'):
        for attr in ['language_model', 'llm', 'decoder']:
            if hasattr(model.model, attr):
                decoder = getattr(model.model, attr)
                break
    
    tokenizer = processor.tokenizer if processor else None
    
    return decoder, tokenizer


def run_decoder_generation(
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 256,
    **generation_kwargs
) -> str | UniformError:
    """
    Run generation using EarthDial's shared Qwen decoder.
    
    Args:
        input_embeds: Fused embeddings (B, seq_len, hidden_dim)
        attention_mask: Optional attention mask
        max_new_tokens: Max tokens to generate
        
    Returns:
        Generated text string or UniformError
    """
    decoder, tokenizer = get_shared_decoder()
    
    if decoder is None or tokenizer is None:
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message="EarthDial decoder not available")
    
    device = next(decoder.parameters()).device
    input_embeds = input_embeds.to(device)
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    try:
        with torch.no_grad():
            output_ids = decoder.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        # Decode only new tokens
        input_len = input_embeds.shape[1]
        new_tokens = output_ids[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.error(f"Decoder generation failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message=f"Decoder generation failed: {e}")


# =============================================================================
# HIGH-LEVEL OPTICAL TASKS (caption, VQA) - using EarthDial directly
# =============================================================================

def run_optical_caption(image_path: str) -> str | UniformError:
    """Run captioning on optical image using EarthDial."""
    ed_config = _get_earthdial_config()
    if ed_config.get("mock", True):
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message="EarthDial mock mode enabled")
    
    processor = get_earthdial_processor()
    model = get_earthdial_model()
    
    if processor is None or model is None:
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message="EarthDial not loaded")
    
    try:
        # Load and preprocess image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Use EarthDial's chat template for captioning
        prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nPlease describe this remote sensing image in detail, including land cover, notable features, and any visible structures. ASSISTANT:"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.error(f"Optical caption failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message=f"Optical caption failed: {e}")


def run_optical_vqa(image_path: str, question: str) -> str | UniformError:
    """Run VQA on optical image using EarthDial."""
    ed_config = _get_earthdial_config()
    if ed_config.get("mock", True):
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message="EarthDial mock mode enabled")
    
    processor = get_earthdial_processor()
    model = get_earthdial_model()
    
    if processor is None or model is None:
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message="EarthDial not loaded")
    
    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.error(f"Optical VQA failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                           message=f"Optical VQA failed: {e}")