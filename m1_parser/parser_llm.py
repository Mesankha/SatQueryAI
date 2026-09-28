"""
Claude LLM parser with structured JSON output.
Primary path: function calling / JSON mode.
"""
import os
import threading
import queue
from typing import Any

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from shared.schemas import StructuredQuery, TaskType
from shared.logger import get_logger

logger = get_logger(__name__)

# Lock to serialize model access; prevents timed-out straggler from overlapping next call
_generation_lock = threading.Lock()

SYSTEM_PROMPT = """You are a satellite imagery query parser. Extract structured fields from the user's natural language query and return valid JSON matching this exact schema:

{
  "query_text": "<original query>",
  "task_type": "search | vqa | caption | change | fusion",
  "location": "<place name or null>",
  "aoi": null,
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "sensor": "Sentinel-1 | Sentinel-2 | both | null",
  "object": "<land cover object or null>",
  "event": "<described event or null>",
  "change_flag": true|false,
  "question": "<verbatim VQA question or null>",
  "referring_expression": "<grounding phrase or null>",
  "confidence": 0.0-1.0,
  "used_fallback": false
}

Rules:
- Question words present → task_type: "vqa"
- "describe", "caption", "what is the land cover", "highlight", "where is", "locate", "show me the" → task_type: "caption"
- "changed", "before and after", "increased", "decreased" → task_type: "change"
- "both", "optical and SAR", "combine", "fusion" → task_type: "fusion"
- Otherwise → task_type: "search"

Use ISO 8601 dates (YYYY-MM-DD). If only a year or month is given, use the first day.
Set confidence based on how clearly the query maps to the schema."""

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Show me Sentinel-2 images of agriculture near Delhi from June 2024"},
    {"role": "assistant", "content": '{"query_text":"Show me Sentinel-2 images of agriculture near Delhi from June 2024","task_type":"search","location":"Delhi","aoi":null,"start_date":"2024-06-01","end_date":"2024-06-30","sensor":"Sentinel-2","object":"agriculture","event":null,"change_flag":false,"question":null,"referring_expression":null,"confidence":0.95,"used_fallback":false}'},
    {"role": "user", "content": "What is the land cover in this image?"},
    {"role": "assistant", "content": '{"query_text":"What is the land cover in this image?","task_type":"vqa","location":null,"aoi":null,"start_date":null,"end_date":null,"sensor":null,"object":null,"event":null,"change_flag":false,"question":"What is the land cover in this image?","referring_expression":null,"confidence":0.92,"used_fallback":false}'},
    {"role": "user", "content": "Describe the land cover around Guwahati"},
    {"role": "assistant", "content": '{"query_text":"Describe the land cover around Guwahati","task_type":"caption","location":"Guwahati","aoi":null,"start_date":null,"end_date":null,"sensor":null,"object":null,"event":null,"change_flag":false,"question":null,"referring_expression":null,"confidence":0.88,"used_fallback":false}'},
    {"role": "user", "content": "Highlight the water body in the north east"},
    {"role": "assistant", "content": '{"query_text":"Highlight the water body in the north east","task_type":"caption","location":null,"aoi":null,"start_date":null,"end_date":null,"sensor":null,"object":"water","event":null,"change_flag":false,"question":null,"referring_expression":"the water body in the north east","confidence":0.9,"used_fallback":false}'},
    {"role": "user", "content": "What changed between August 2023 and February 2024 in Assam?"},
    {"role": "assistant", "content": '{"query_text":"What changed between August 2023 and February 2024 in Assam?","task_type":"change","location":"Assam","aoi":null,"start_date":"2023-08-01","end_date":"2024-02-29","sensor":null,"object":null,"event":"flooding","change_flag":true,"question":null,"referring_expression":null,"confidence":0.85,"used_fallback":false}'},
    {"role": "user", "content": "Combine optical and SAR for the forest area"},
    {"role": "assistant", "content": '{"query_text":"Combine optical and SAR for the forest area","task_type":"fusion","location":null,"aoi":null,"start_date":null,"end_date":null,"sensor":"both","object":"forest","event":null,"change_flag":false,"question":null,"referring_expression":null,"confidence":0.8,"used_fallback":false}'},
]


# Lazy-loaded pipeline singleton
_pipeline = None

def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    
    if not torch.cuda.is_available():
        logger.warning("M1: CUDA unavailable, falling back (local LLM requires GPU)")
        return None
        
    model_id = "microsoft/Phi-4-mini-instruct"
    logger.info(f"M1: Loading {model_id} in 4-bit mode... This may take a moment.")
    
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        _pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            return_full_text=False
        )
        return _pipeline
    except Exception as exc:
        logger.warning(f"M1: Failed to load local LLM: {exc}")
        return None


def _generate_once(pipe: Any, messages: list[dict]) -> dict | None:
    """
    Run a single generation pass. Must hold _generation_lock for the entire duration.
    Returns parsed JSON dict or None on failure.
    """
    import json as _json
    
    try:
        outputs = pipe(
            messages,
            max_new_tokens=512,
            do_sample=False,  # deterministic
            temperature=0.0,
        )
        
        raw = outputs[0]["generated_text"].strip()
        
        # Extract JSON if wrapped in markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
            
        data = _json.loads(raw.strip())
        return data
    except Exception as exc:
        logger.warning(f"M1: LLM generation failed: {exc}")
        return None


def _generate_with_timeout(pipe: Any, messages: list[dict], timeout_seconds: float) -> dict | None:
    """
    Run _generate_once in a daemon thread with a timeout.
    Returns parsed JSON dict or None if timeout or failure.
    """
    result_queue: queue.Queue[dict | None] = queue.Queue(maxsize=1)
    
    def worker():
        # Hold lock for the entire generation to prevent overlap
        with _generation_lock:
            result = _generate_once(pipe, messages)
            result_queue.put(result)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running - timeout occurred
        logger.warning(f"M1: LLM generation timed out after {timeout_seconds}s")
        return None
    
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None


def parse_llm(query_text: str, timeout_seconds: float = 30.0) -> StructuredQuery | None:
    """Call local Phi-4-mini to parse query. Returns None on any failure so caller can fallback."""
    pipe = _get_pipeline()
    if pipe is None:
        return None

    messages = list(FEW_SHOT_EXAMPLES)
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": query_text})

    data = _generate_with_timeout(pipe, messages, timeout_seconds)
    if data is None:
        return None
    
    try:
        return StructuredQuery.model_validate(data)
    except Exception as exc:
        logger.warning(f"M1: LLM parse validation failed: {exc}")
        return None


def parse_with_llm(prompt: str, timeout_seconds: float = 30.0) -> dict | None:
    """
    Generic LLM call for synthesis tasks.
    Returns parsed JSON dict or None on failure.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return None

    messages = [
        {"role": "system", "content": "You are a remote sensing analyst. Return valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    data = _generate_with_timeout(pipe, messages, timeout_seconds)
    if data is None:
        return None
    
    # Ensure it's a dict with expected keys
    if isinstance(data, dict):
        return data
    return None
