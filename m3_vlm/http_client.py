"""
M3 VLM HTTP Client for M5 Integration
Provides async HTTP client to call M3 Flask service (port 8001).
Also provides mock responses for offline demo mode.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
import httpx

from shared.schemas import UniformError, ErrorType
from shared.config import get_config

logger = logging.getLogger(__name__)


def _is_mock_mode(modality: str = "optical") -> bool:
    """Check if mock mode is enabled for the given modality."""
    config = get_config()
    vlm_config = config.get("vlm", {})
    if modality == "sar":
        return vlm_config.get("sar_mock", True)
    return vlm_config.get("mock", True)


def _mock_vqa_response(image_path: str, question: str) -> Dict[str, Any]:
    """Generate mock VQA response."""
    return {
        "answer": f"Mock VQA answer for {image_path}: {question}",
        "fallback_used": True,
    }


def _mock_caption_response(image_path: str) -> Dict[str, Any]:
    """Generate mock caption response."""
    return {
        "caption": f"Mock caption for {image_path}: This is a simulated satellite image showing typical remote sensing features including land cover, water bodies, and vegetation patterns.",
        "fallback_used": True,
    }


def _mock_grounding_response(image_path: str, expression: str) -> Dict[str, Any]:
    """Generate mock grounding response."""
    return {
        "bbox": {"x1": 0.2, "y1": 0.3, "x2": 0.7, "y2": 0.8},
        "fallback_used": True,
    }


class M3VLMClient:
    """Async HTTP client for M3 VLM service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        modality: str = "optical",
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic, or return mock response in mock mode."""
        # In mock mode, return mock responses immediately without HTTP calls
        if _is_mock_mode(modality):
            if endpoint == "/vqa":
                return _mock_vqa_response(json_data.get("image_path", ""), json_data.get("question", ""))
            elif endpoint == "/caption":
                return _mock_caption_response(json_data.get("image_path", ""))
            elif endpoint == "/grounding":
                return _mock_grounding_response(json_data.get("image_path", ""), json_data.get("expression", ""))
            elif endpoint == "/health":
                return {"status": "ok", "mode": "mock"}

        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=json_data)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as e:
                last_error = f"Request timeout after {self.timeout}s"
                logger.warning(f"M3 request timeout (attempt {attempt + 1}/{self.max_retries + 1}): {url}")
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.warning(f"M3 HTTP error (attempt {attempt + 1}/{self.max_retries + 1}): {last_error}")
                if e.response.status_code < 500:
                    # Don't retry client errors
                    break
            except httpx.RequestError as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"M3 connection error (attempt {attempt + 1}/{self.max_retries + 1}): {last_error}")
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"M3 unexpected error: {last_error}")
                break

            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # All retries failed
        return self._make_error("model_unavailable", last_error or "M3 service unavailable")

    def _make_error(self, error_type: str, message: str) -> Dict[str, Any]:
        return {
            "error": True,
            "error_type": error_type,
            "message": message,
            "fallback_used": False,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check M3 service health."""
        return await self._request_with_retry("GET", "/health")

    async def vqa(self, image_path: str, question: str, modality: str = "optical") -> Dict[str, Any]:
        """Visual Question Answering."""
        return await self._request_with_retry("POST", "/vqa", {
            "image_path": image_path,
            "question": question,
        }, modality=modality)

    async def caption(self, image_path: str, modality: str = "optical") -> Dict[str, Any]:
        """Image Captioning."""
        return await self._request_with_retry("POST", "/caption", {
            "image_path": image_path,
        }, modality=modality)

    async def grounding(self, image_path: str, expression: str) -> Dict[str, Any]:
        """Referring Expression Grounding."""
        return await self._request_with_retry("POST", "/grounding", {
            "image_path": image_path,
            "expression": expression,
        })


# Global client instance
_client: Optional[M3VLMClient] = None


def get_m3_client(config: Optional[Dict[str, Any]] = None) -> M3VLMClient:
    """Get or create global M3 client."""
    global _client
    if _client is None:
        base_url = "http://localhost:8001"
        timeout = 30.0
        if config:
            base_url = config.get("base_url", base_url)
            timeout = config.get("timeout", timeout)
        _client = M3VLMClient(base_url=base_url, timeout=timeout)
    return _client


async def close_m3_client():
    """Close global M3 client."""
    global _client
    if _client:
        await _client.close()
        _client = None


# Convenience functions for backward compatibility
async def run_vqa_http(image_path: str, question: str, modality: str = "optical") -> Dict[str, Any]:
    """Run VQA via HTTP."""
    client = get_m3_client()
    return await client.vqa(image_path, question, modality=modality)


async def run_caption_http(image_path: str, modality: str = "optical") -> Dict[str, Any]:
    """Run caption via HTTP."""
    client = get_m3_client()
    return await client.caption(image_path, modality=modality)


async def run_grounding_http(image_path: str, expression: str) -> Dict[str, Any]:
    """Run grounding via HTTP."""
    client = get_m3_client()
    return await client.grounding(image_path, expression)