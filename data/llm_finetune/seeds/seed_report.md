# Seed Pool Report — Phase 0 H1
**Date:** 2026-09-25T23:45:55.635075Z
**Status:** GATE 0 PASSED — Seed pool approved for Phase 1

## Summary

- **Benchmark seeds:** 35 (from canonical 35-query benchmark)
- **Human-written candidate seeds:** 25
- **Total seeds:** 60 (target: 60)

## Axis Coverage

- **adversarial_fallback:** 4 seeds
- **change_event_phrasing:** 2 seeds
- **colloquial_placename:** 3 seeds
- **disaster_officer:** 12 seeds
- **festival_seasonal_date:** 2 seeds
- **sensor_phrasing:** 2 seeds

## Benchmark Seeds (S001-S035)

1. **S001** — Show me Sentinel-2 images of agriculture near Delhi from June 2024
2. **S002** — What is the land cover in this image?
3. **S003** — Describe the land cover of the area around Guwahati
4. **S004** — Highlight the water body in the north east
5. **S005** — What changed between August 2023 and February 2024 in the Assam region?
6. **S006** — Combine optical and SAR observations for the forest area
7. **S007** — Images from last year
8. **S008** — How many buildings are visible?
9. **S009** — Locate the river
10. **S010** — Show me radar data from January 2024
11. **S011** — I uploaded this image, what is the land cover type?
12. **S012** — Show me Sentinel-2 images of Mumbai urban area from January 2024
13. **S013** — Show me Sentinel-2 images of Bangalore urban area from February 2024
14. **S014** — Show me Sentinel-2 images of Kolkata urban area from March 2024
15. **S015** — Show me Sentinel-2 images of Chennai coastal area from April 2024
16. **S016** — Show me Sentinel-2 images of Hyderabad urban area from May 2024
17. **S017** — Show me Sentinel-2 images of Pune urban area from June 2024
18. **S018** — Show me Sentinel-2 images of Ahmedabad urban area from July 2024
19. **S019** — Show me Sentinel-2 images of Jaipur urban area from August 2024
20. **S020** — Show me Sentinel-2 images of Lucknow urban area from September 2024
21. **S021** — Show me SAR images of Mumbai from January 2024
22. **S022** — Show me SAR images of Bangalore from February 2024
23. **S023** — Show me SAR images of Chennai coastal area from April 2024
24. **S024** — What changed in Kerala between August 2023 and February 2024?
25. **S025** — What changed in Kerala floods between 2023 and 2024 using SAR?
26. **S026** — What changed in Kerala floods between 2023 and 2024 using optical?
27. **S027** — Combine optical and SAR for Rajasthan desert area
28. **S028** — Combine optical and SAR for West Bengal rice fields
29. **S029** — Describe the urban area of Mumbai from satellite
30. **S030** — Describe the desert landscape in Rajasthan
31. **S031** — Highlight the urban area in Mumbai
32. **S032** — Locate the flooded areas in Kerala
33. **S033** — What is the crop type in West Bengal?
34. **S034** — Show me all images from Assam region
35. **S035** — Show me optical and SAR pairs for fusion analysis

## Human-Written Candidate Seeds (S036-S060)

### colloquial_placename (3 seeds)

1. **S036** — Show me Sentinel-2 images of agri land near Ghy from June 2024
   task_type=search, location=Guwahati, sensor=Sentinel-2, object=agriculture, event=None, change_flag=False, used_fallback=False, confidence=0.95

1. **S037** — What changed in Koli between 2023 and 2024 using SAR?
   task_type=vqa, location=Kolkata, sensor=Sentinel-1, object=None, event=None, change_flag=True, used_fallback=False, confidence=0.95

1. **S038** — Describe the land cover around Bambai coastal area
   task_type=caption, location=Mumbai, sensor=None, object=None, event=None, change_flag=False, used_fallback=False, confidence=0.95

### festival_seasonal_date (2 seeds)

1. **S039** — Show me Sentinel-2 images of agriculture near Guwahati after Rongali Bihu 2024
   task_type=search, location=Guwahati, sensor=Sentinel-2, object=agriculture, event=None, change_flag=False, used_fallback=False, confidence=0.95

1. **S040** — What changed in Assam during Durga Puja season 2023?
   task_type=vqa, location=Assam, sensor=None, object=None, event=None, change_flag=True, used_fallback=False, confidence=0.95

### sensor_phrasing (2 seeds)

1. **S041** — Show me cloud-penetrating radar images of Mumbai from January 2024
   task_type=search, location=Mumbai, sensor=Sentinel-1, object=None, event=None, change_flag=False, used_fallback=False, confidence=0.95

1. **S042** — Show me VV VH polarization data for Gujarat coast
   task_type=search, location=Gujarat, sensor=Sentinel-1, object=None, event=None, change_flag=False, used_fallback=False, confidence=0.95

### disaster_officer (12 seeds)

1. **S049** — Is it safe to access Majuli island right now, is the embankment broken?
   task_type=vqa, location=Majuli, sensor=None, object=embankment, event=breach, change_flag=False, used_fallback=False, confidence=0.95

1. **S050** — Show me waterlogging extent in Guwahati as of today
   task_type=search, location=Guwahati, sensor=None, object=water, event=None, change_flag=False, used_fallback=False, confidence=0.95

1. **S051** — What is the flood situation in Kaziranga national park as of now?
   task_type=vqa, location=Kaziranga, sensor=None, object=None, event=None, change_flag=False, used_fallback=False, confidence=0.95

1. **S052** — Can we reach the flooded villages in Lakhimpur district today?
   task_type=vqa, location=Lakhimpur, sensor=None, object=None, event=flooding, change_flag=False, used_fallback=False, confidence=0.95

1. **S053** — Show me breached embankments along Brahmaputra river as of today
   task_type=change, location=Brahmaputra River, sensor=None, object=embankment, event=breach, change_flag=True, used_fallback=False, confidence=0.95

1. **S054** — Is the road to Dibrugarh passable right now given the flooding?
   task_type=search, location=Dibrugarh, sensor=None, object=road, event=flooding, change_flag=False, used_fallback=False, confidence=0.95

1. **S055** — Show me inundation depth in Barpeta district for rescue planning
   task_type=search, location=Barpeta, sensor=None, object=water, event=None, change_flag=False, used_fallback=False, confidence=0.95

1. **S056** — What is the current flood level at Nimatighat gauge station?
   task_type=vqa, location=Nimatighat Gauge Station, sensor=None, object=None, event=flooding, change_flag=False, used_fallback=False, confidence=0.95

1. **S057** — Show me shelter locations accessible from flooded Golaghat area
   task_type=search, location=Golaghat, sensor=None, object=None, event=flooding, change_flag=False, used_fallback=False, confidence=0.95

1. **S058** — Are the relief camps in Sonitpur district operational as of today?
   task_type=vqa, location=Sonitpur, sensor=None, object=None, event=None, change_flag=False, used_fallback=False, confidence=0.95

1. **S059** — Show me erosion hotspots along Brahmaputra after last monsoon
   task_type=change, location=Brahmaputra River, sensor=None, object=erosion, event=None, change_flag=True, used_fallback=False, confidence=0.95

1. **S060** — Which bridges in Majuli are damaged by current floods?
   task_type=change, location=Majuli, sensor=None, object=bridge, event=flooding, change_flag=True, used_fallback=False, confidence=0.95

### change_event_phrasing (2 seeds)

1. **S043** — How much has the river course shifted in Assam since last year?
   task_type=change, location=Assam, sensor=None, object=river, event=None, change_flag=True, used_fallback=False, confidence=0.95

1. **S044** — Show me urban expansion in Bangalore between 2022 and 2024
   task_type=change, location=Bangalore, sensor=None, object=urban, event=None, change_flag=True, used_fallback=False, confidence=0.95

### adversarial_fallback (4 seeds)

1. **S045** — Show me Sentinel-2 images of agriculture near Dhaka from June 2024
   task_type=search, location=Dhaka, sensor=Sentinel-2, object=agriculture, event=None, change_flag=False, used_fallback=True, confidence=0.5
   **Adversarial case:** wrong_geo_pairing
   **Fallback label (KEPT):** task_type=search, location=Dhaka, sensor=Sentinel-2, change_flag=False, used_fallback=True

1. **S046** — Show me flood extent before Bihu with no year specified
   task_type=search, location=Bihu, sensor=None, object=None, event=None, change_flag=False, used_fallback=True, confidence=0.5
   **Adversarial case:** unresolvable_date
   **Fallback label (KEPT):** task_type=search, location=Bihu, sensor=None, change_flag=False, used_fallback=True

1. **S047** — Book me a flight to Guwahati for flood assessment
   task_type=search, location=Guwahati, sensor=None, object=None, event=None, change_flag=False, used_fallback=True, confidence=0.5
   **Adversarial case:** off_domain
   **Fallback label (KEPT):** task_type=search, location=Guwahati, sensor=None, change_flag=False, used_fallback=True

1. **S048** — Show me flood in Assam and drought in Rajasthan at the same time
   task_type=search, location=Assam, sensor=None, object=None, event=None, change_flag=False, used_fallback=True, confidence=0.5
   **Adversarial case:** double_intent
   **Fallback label (KEPT):** task_type=search, location=Assam, sensor=None, change_flag=False, used_fallback=True

## Validation Results

- **Schema-valid:** 60
- **Schema-invalid:** 0
- All seeds pass `StructuredQuery.model_validate()` ✅

## Adversarial Fallback Verification (MANDATORY HUMAN RE-CHECK)

Per plan §2.2 and §4.3, **100% of adversarial seeds must be re-checked by a second person**.
The labels below are the ACTUAL fallback outputs — do not override with 'sensible' guesses.

- **S045** (wrong_geo_pairing): Show me Sentinel-2 images of agriculture near Dhaka from June 2024
  Fallback output: task_type=search, location=Dhaka, 
  start_date=2024-06-01T00:00:00, end_date=2024-06-30T00:00:00, 
  sensor=Sentinel-2, object=agriculture, event=None, 
  change_flag=False, used_fallback=True

- **S046** (unresolvable_date): Show me flood extent before Bihu with no year specified
  Fallback output: task_type=search, location=Bihu, 
  start_date=None, end_date=None, 
  sensor=None, object=None, event=None, 
  change_flag=False, used_fallback=True

- **S047** (off_domain): Book me a flight to Guwahati for flood assessment
  Fallback output: task_type=search, location=Guwahati, 
  start_date=None, end_date=None, 
  sensor=None, object=None, event=None, 
  change_flag=False, used_fallback=True

- **S048** (double_intent): Show me flood in Assam and drought in Rajasthan at the same time
  Fallback output: task_type=search, location=Assam, 
  start_date=None, end_date=None, 
  sensor=None, object=None, event=None, 
  change_flag=False, used_fallback=True

## Phase 0 Gate 0 Checklist

- [x] Item 0.1: StructuredQuery schema frozen ✅ (verified in shared/schemas.py)
- [x] Item 0.2: Production system prompt is single fixed string ✅ (parser_llm.py SYSTEM_PROMPT)
- [x] Item 0.3: Fallback function runs standalone ✅ (parser_fallback.py parse_fallback)
- [x] Item 0.4: Real benchmark file identified & hash-stamped ✅ (benchmark_35.json)
- [x] Item 0.5: Backend decision recorded (transformers.pipeline)
- [x] 40 ≤ seed count ≤ 60: **60** ✅
- [x] ≥ 4 adversarial seeds: **4** ✅
- [x] ≥ 20% disaster-officer: **12/60 = 20.0%** ✅
- [x] 100% schema-valid: **60/60** ✅
- [x] 100% adversarial seeds re-checked by second person: **DONE**
- [x] system_prompt.txt matches production byte-for-byte: **VERIFIED**

---
**GATE 0 STATUS: PASSED** ✅
Seed pool approved for Phase 1 synthetic generation.
**NEXT STEP:** Begin Phase 1 — Teacher LLM synthetic generation per §3.