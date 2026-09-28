"""
M3 VLM Runner - Lazy singleton for GeoChat 4-bit inference.
Model weights load only when vlm.mock=false in config.
"""
import os
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional, Dict, Any

from shared.schemas import UniformError, ErrorType
from shared.config import get_config

logger = logging.getLogger(__name__)

# Global state for lazy singleton
_vlm_model = None
_vlm_processor = None
_vlm_executor: Optional[ThreadPoolExecutor] = None
_vlm_lock = threading.Lock()
_vlm_initialized = False
_vlm_init_error: Optional[str] = None


def _get_vlm_config() -> Dict[str, Any]:
    """Get VLM configuration from config.yaml."""
    config = get_config()
    return config.get("vlm", {})


def load_vlm() -> None:
    """Lazy singleton loader for GeoChat 4-bit model.
    
    Loads model only when vlm.mock=false. Uses bitsandbytes 4-bit quantization.
    Device: CUDA if available, else raises UniformError on first use.
    """
    global _vlm_model, _vlm_processor, _vlm_executor, _vlm_initialized, _vlm_init_error
    
    with _vlm_lock:
        if _vlm_initialized:
            return
        
        vlm_config = _get_vlm_config()
        if vlm_config.get("mock", True):
            logger.info("VLM mock mode enabled, skipping model load")
            _vlm_initialized = True
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available - VLM requires GPU")
            
            model_path = vlm_config.get("model_path", "mbzuai-oryx/GeoChat")
            load_in_4bit = vlm_config.get("load_in_4bit", True)
            lora_adapter_path = vlm_config.get("lora_adapter_path")
            
            logger.info(f"Loading VLM from {model_path} (4-bit={load_in_4bit})")
            
            compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                _vlm_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map={"": 0},
                    trust_remote_code=True,
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                _vlm_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map={"": 0},
                    trust_remote_code=True,
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                )
            
            _vlm_processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # Set pad token
            if _vlm_processor.tokenizer.pad_token is None:
                tok = _vlm_processor.tokenizer
                if tok.unk_token is not None:
                    tok.pad_token = tok.unk_token
                else:
                    tok.add_special_tokens({"pad_token": "<pad>"})
                    _vlm_model.resize_token_embeddings(len(tok))
            
            # Load LoRA adapter if present
            if lora_adapter_path and os.path.exists(lora_adapter_path):
                logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
                from peft import PeftModel
                _vlm_model = PeftModel.from_pretrained(_vlm_model, lora_adapter_path, is_trainable=False)
            
            _vlm_model.eval()
            _vlm_executor = ThreadPoolExecutor(max_workers=1)
            _vlm_initialized = True
            logger.info("VLM loaded successfully")
            
        except Exception as e:
            _vlm_init_error = str(e)
            logger.error(f"VLM load failed: {e}")
            _vlm_initialized = True  # Mark as initialized to avoid retry loops
            raise


def unload_vlm() -> None:
    """Free GPU memory and cleanup."""
    global _vlm_model, _vlm_processor, _vlm_executor, _vlm_initialized, _vlm_init_error
    
    with _vlm_lock:
        if _vlm_executor:
            _vlm_executor.shutdown(wait=True)
            _vlm_executor = None
        if _vlm_model is not None:
            _vlm_model.cpu()
            del _vlm_model
            _vlm_model = None
        if _vlm_processor is not None:
            del _vlm_processor
            _vlm_processor = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _vlm_initialized = False
        _vlm_init_error = None
        logger.info("VLM unloaded")


def _run_inference(prompt: str, image_path: str, max_new_tokens: int = 256) -> str:
    """Run VLM inference on the dedicated executor thread."""
    import torch
    
    # Load and preprocess image
    from m3_vlm.vlm_service import _load_image_safely
    image = _load_image_safely(image_path, modality="auto")
    
    # Prepare inputs
    inputs = _vlm_processor(text=prompt, images=image, return_tensors="pt").to(_vlm_model.device)
    
    # Generate
    with torch.no_grad():
        output_ids = _vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=_vlm_processor.tokenizer.pad_token_id,
            eos_token_id=_vlm_processor.tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    response = _vlm_processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def _run_modality_inference(
    prompt_template: str,
    image_path: str,
    max_new_tokens: int = 256,
    **format_kwargs
) -> str:
    """Run inference with modality-specific prompt template."""
    prompt = prompt_template.format(**format_kwargs)
    return _run_inference(prompt, image_path, max_new_tokens)


def _ensure_vlm_loaded(vlm_config: Dict[str, Any], mock_key: str = "mock") -> Optional[UniformError]:
    """Ensure VLM is loaded. Returns UniformError if mock mode or load failed."""
    if vlm_config.get(mock_key, True):
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, 
                           message=f"VLM {mock_key} mode enabled")
    
    if not _vlm_initialized:
        try:
            load_vlm()
        except Exception as e:
            return UniformError(module="M3", error_type=ErrorType.model_unavailable, 
                               message=f"VLM load failed: {e}")
    
    if _vlm_model is None or _vlm_executor is None:
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, 
                           message="VLM not available")
    
    return None


def run_caption(image_path: str) -> str | UniformError:
    """Run captioning on image. Returns caption string or UniformError."""
    vlm_config = _get_vlm_config()
    err = _ensure_vlm_loaded(vlm_config, "mock")
    if err:
        return err
    
    timeout = vlm_config.get("inference_timeout", 30)
    
    # Prompt template for captioning
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: <image>\nPlease describe this remote sensing image in detail, "
        "including land cover, notable features, and any visible structures. ASSISTANT:"
    )
    
    future: Future = _vlm_executor.submit(_run_inference, prompt, image_path, 256)
    try:
        result = future.result(timeout=timeout)
        return result
    except Exception as e:
        logger.error(f"Caption inference failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, message=str(e))


def run_vqa(image_path: str, question: str) -> str | UniformError:
    """Run VQA on image with question. Returns answer string or UniformError."""
    vlm_config = _get_vlm_config()
    err = _ensure_vlm_loaded(vlm_config, "mock")
    if err:
        return err
    
    timeout = vlm_config.get("inference_timeout", 30)
    
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: <image>\n{question} ASSISTANT:"
    )
    
    future: Future = _vlm_executor.submit(_run_inference, prompt, image_path, 256)
    try:
        result = future.result(timeout=timeout)
        return result
    except Exception as e:
        logger.error(f"VQA inference failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, message=str(e))


def run_sar_caption(image_path: str) -> str | UniformError:
    """Run SAR captioning with SAR-specific prompt."""
    vlm_config = _get_vlm_config()
    err = _ensure_vlm_loaded(vlm_config, "sar_mock")
    if err:
        return err
    
    timeout = vlm_config.get("inference_timeout", 30)
    
    # Use SAR-specific prompt from sar_prompts
    from m3_vlm.sar_prompts import select_prompt
    prompt = select_prompt("caption", "sar")
    
    future: Future = _vlm_executor.submit(_run_inference, prompt, image_path, 256)
    try:
        result = future.result(timeout=timeout)
        return result
    except Exception as e:
        logger.error(f"SAR Caption inference failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, message=str(e))


def run_sar_vqa(image_path: str, question: str) -> str | UniformError:
    """Run SAR VQA with SAR-specific prompt."""
    vlm_config = _get_vlm_config()
    err = _ensure_vlm_loaded(vlm_config, "sar_mock")
    if err:
        return err
    
    timeout = vlm_config.get("inference_timeout", 30)
    
    from m3_vlm.sar_prompts import select_prompt
    prompt = select_prompt("vqa", "sar", question=question)
    
    future: Future = _vlm_executor.submit(_run_inference, prompt, image_path, 256)
    try:
        result = future.result(timeout=timeout)
        return result
    except Exception as e:
        logger.error(f"SAR VQA inference failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, message=str(e))


def run_vqa(image_path: str, question: str) -> str | UniformError:
    """Run VQA on image with question. Returns answer string or UniformError."""
    vlm_config = _get_vlm_config()
    if vlm_config.get("mock", True):
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, message="VLM mock mode enabled")
    
    if not _vlm_initialized:
        try:
            load_vlm()
        except Exception as e:
            return UniformError(module="M3", error_type=ErrorType.model_unavailable, message=f"VLM load failed: {e}")
    
    if _vlm_model is None or _vlm_executor is None:
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, message="VLM not available")
    
    timeout = vlm_config.get("inference_timeout", 30)
    
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: <image>\n{question} ASSISTANT:"
    )
    
    future: Future = _vlm_executor.submit(_run_inference, prompt, image_path, 256)
    try:
        result = future.result(timeout=timeout)
        return result
    except Exception as e:
        logger.error(f"VQA inference failed: {e}")
        return UniformError(module="M3", error_type=ErrorType.model_unavailable, message=str(e))