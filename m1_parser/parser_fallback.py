"""
Deterministic fallback parser — pure Python, zero ML, zero API calls.
O(n) string scanning, <10ms latency, 100% reproducible.
"""
import re
from datetime import datetime, timedelta
from typing import Any

try:
    import dateparser
except ImportError:  # pragma: no cover
    dateparser = None

from shared.schemas import StructuredQuery, TaskType, Sensor, AOI


# Keyword → task_type mapping
TASK_KEYWORDS = {
    TaskType.vqa: ["what", "how many", "where is", "when did", "why", "which"],
    TaskType.caption: ["describe", "caption", "what is the land cover", "what does this show",
                       "highlight", "where is", "locate", "show me the", "find the", "point to"],
    TaskType.change: ["changed", "before and after", "increased", "decreased", "difference between", "what changed"],
    TaskType.fusion: ["both", "optical and sar", "combine", "fusion", "multimodal", "sar and optical"],
    TaskType.sar_change: ["sar change", "radar change", "sar difference", "radar difference"],
    TaskType.sar_grounding: ["ground on sar", "locate on sar", "find on sar", "highlight on sar"],
}

SENSOR_KEYWORDS = {
    Sensor.sentinel_1: ["sar", "radar", "sentinel-1", "sentinel 1", "vv", "vh"],
    Sensor.sentinel_2: ["optical", "multispectral", "sentinel-2", "sentinel 2", "rgb", "ndvi"],
}


def _infer_task_type(text: str) -> TaskType:
    text_lower = text.lower()
    scores = {task: 0 for task in TaskType}
    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[task] += 1
    # Priority order: specific SAR tasks > fusion > change > vqa > caption > search
    if scores[TaskType.sar_grounding] > 0:
        return TaskType.sar_grounding
    if scores[TaskType.sar_change] > 0:
        return TaskType.sar_change
    if scores[TaskType.fusion] > 0:
        return TaskType.fusion
    if scores[TaskType.change] > 0:
        return TaskType.change
    if scores[TaskType.vqa] > 0:
        return TaskType.vqa
    if scores[TaskType.caption] > 0:
        return TaskType.caption
    return TaskType.search


def _infer_sensor(text: str) -> Sensor | None:
    text_lower = text.lower()
    s1_hits = sum(1 for kw in SENSOR_KEYWORDS[Sensor.sentinel_1] if kw in text_lower)
    s2_hits = sum(1 for kw in SENSOR_KEYWORDS[Sensor.sentinel_2] if kw in text_lower)
    if s1_hits > 0 and s2_hits > 0:
        return Sensor.both
    if s1_hits > 0:
        return Sensor.sentinel_1
    if s2_hits > 0:
        return Sensor.sentinel_2
    return None


def _month_name_to_number(name: str) -> int | None:
    """Convert month name (full or 3-letter abbreviation) to 1-12."""
    _MONTHS = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    return _MONTHS.get(name.lower().strip())


def _month_year_to_range(month: int, year: int) -> tuple[datetime, datetime]:
    """Convert month+year to (first-day, last-day) datetime range."""
    import calendar
    _, last_day = calendar.monthrange(year, month)
    return datetime(year, month, 1), datetime(year, month, last_day)


def _extract_dates(text: str) -> tuple[datetime | None, datetime | None]:
    """Extract date range using regex patterns and dateparser fallback.

    Handles (in priority order):
      1. ISO dates: 2023-08-01
      2. Month Year pairs: "August 2023", "Feb 2024"
      3. Relative phrases: "last year", "last month"
      4. Generic dateparser fallback
    """
    # --- 1. Try ISO dates first ---
    iso_pattern = r"(\d{4}-\d{2}-\d{2})"
    iso_matches = re.findall(iso_pattern, text)
    if len(iso_matches) >= 2:
        return datetime.strptime(iso_matches[0], "%Y-%m-%d"), datetime.strptime(iso_matches[1], "%Y-%m-%d")
    if len(iso_matches) == 1:
        d = datetime.strptime(iso_matches[0], "%Y-%m-%d")
        return d, d

    # --- 2. Try "Month Year" patterns (e.g. "August 2023", "Feb 2024") ---
    month_year_pattern = (
        r"(January|February|March|April|May|June|July|August|September|October|November|December"
        r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+(\d{4})"
    )
    my_matches = re.findall(month_year_pattern, text, re.IGNORECASE)
    if len(my_matches) >= 2:
        m1 = _month_name_to_number(my_matches[0][0])
        y1 = int(my_matches[0][1])
        m2 = _month_name_to_number(my_matches[1][0])
        y2 = int(my_matches[1][1])
        if m1 and m2:
            start, _ = _month_year_to_range(m1, y1)
            _, end = _month_year_to_range(m2, y2)
            return start, end
    if len(my_matches) == 1:
        m1 = _month_name_to_number(my_matches[0][0])
        y1 = int(my_matches[0][1])
        if m1:
            start, end = _month_year_to_range(m1, y1)
            return start, end

    # --- 3. Try "Year" alone (e.g. "from 2023", "in 2024") ---
    year_pattern = r"(?:from|in|during|year)\s+(\d{4})"
    year_matches = re.findall(year_pattern, text, re.IGNORECASE)
    if year_matches:
        y = int(year_matches[0])
        return datetime(y, 1, 1), datetime(y, 12, 31)

    # --- 4. Relative date heuristics ---
    text_lower = text.lower()
    now = datetime.utcnow()
    if "last year" in text_lower:
        start = datetime(now.year - 1, 1, 1)
        end = datetime(now.year - 1, 12, 31)
        return start, end
    if "last month" in text_lower:
        first_this = now.replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
        return start, end

    # --- 5. Generic dateparser fallback (relaxed) ---
    if dateparser is not None:
        parsed = dateparser.parse(text, settings={"STRICT_PARSING": False})
        if parsed:
            return parsed, parsed

    return None, None


# Common false-positive capitalized words that are NOT locations
_NON_LOCATION_WORDS = {
    "show", "what", "how", "when", "where", "why", "which", "describe",
    "highlight", "locate", "find", "combine", "images", "image",
    "sentinel", "sar", "optical", "ndvi", "rgb", "radar",
    "the", "this", "that", "from", "with", "for", "and", "but",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

# Common land cover / object keywords
_OBJECT_KEYWORDS = [
    "agriculture", "forest", "water", "urban", "building", "buildings",
    "river", "lake", "road", "roads", "vegetation", "cropland",
    "wetland", "grassland", "barren", "snow", "ice", "desert",
    "residential", "industrial", "commercial", "farmland",
]


def _extract_object(text: str) -> str | None:
    """Extract land cover object keyword from text."""
    text_lower = text.lower()
    for kw in _OBJECT_KEYWORDS:
        if kw in text_lower:
            return kw
    return None


def _extract_location(text: str) -> str | None:
    """Simple gazetteer + regex for capitalized place names."""
    # Check for common place-preposition patterns first
    # Only capture the first capitalized word(s) after the preposition, stop at lowercase/stop words
    place_pattern = r"(?:near|around|in|of|over)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"
    m = re.search(place_pattern, text)
    if m:
        candidate = m.group(1).strip()
        # Filter out non-location words
        if candidate.lower().split()[0] not in _NON_LOCATION_WORDS:
            return candidate
    # Capitalized word sequences (not at sentence start, not common words)
    cap_pattern = r"(?<!^)(?<!\. )\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b"
    matches = re.findall(cap_pattern, text)
    filtered = [
        m for m in matches
        if m.lower().split()[0] not in _NON_LOCATION_WORDS
        and not any(w.lower() in _NON_LOCATION_WORDS for w in m.split())
    ]
    if filtered:
        return filtered[-1]
    return None


def _extract_question(text: str) -> str | None:
    text_stripped = text.strip()
    if "?" in text_stripped:
        return text_stripped
    return None


def _extract_referring_expression(text: str) -> str | None:
    text_lower = text.lower()
    triggers = ["highlight", "locate", "show me the", "find the", "point to", "where is the"]
    for t in triggers:
        if t in text_lower:
            idx = text_lower.find(t) + len(t)
            return text[idx:].strip(" .")
    return None


def parse_fallback(query_text: str) -> StructuredQuery:
    """Deterministic parser — always returns a schema-valid StructuredQuery."""
    task_type = _infer_task_type(query_text)
    start_date, end_date = _extract_dates(query_text)
    sensor = _infer_sensor(query_text)
    location = _extract_location(query_text)
    obj = _extract_object(query_text)
    question = _extract_question(query_text) if task_type == TaskType.vqa else None
    # Extract referring expression for caption tasks (grounding keywords now map to caption)
    referring_expression = _extract_referring_expression(query_text) if task_type in (TaskType.caption, TaskType.grounding) else None
    change_flag = task_type == TaskType.change

    return StructuredQuery(
        query_text=query_text,
        task_type=task_type,
        location=location,
        start_date=start_date,
        end_date=end_date,
        sensor=sensor,
        object=obj,
        question=question,
        referring_expression=referring_expression,
        change_flag=change_flag,
        confidence=0.5,
        used_fallback=True,
    )
