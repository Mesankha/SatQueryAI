"""
Pre-cache the 5 signature queries for offline replay.
Run: python precache_queries.py
"""
import asyncio
import hashlib
import uuid
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from m5_controller.dispatch_table import orchestrate_async
from m6_api.main import _cache_response

queries = [
    "Show me SAR images of flooding near Guwahati from August 2023",
    "What changed between August 2023 and February 2024 in Assam?",
    "Describe the land cover around Silchar",
    "Highlight the water body near Majuli",
    "Combine optical and SAR for forest monitoring in Barpeta",
]

def get_deterministic_id(text: str) -> str:
    h = hashlib.md5(text.lower().strip().encode()).hexdigest()
    return str(uuid.UUID(h))

async def main():
    for q in queries:
        print(f"Pre-caching: {q}")
        try:
            response = await orchestrate_async(q)
            # Assign deterministic ID
            q_id = get_deterministic_id(q)
            response.query_id = q_id
            _cache_response(q_id, response)
            print(f"  -> Saved as {q_id}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
