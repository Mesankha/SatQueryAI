# Seed Semantic Audit — Phase 0 H1
**Date:** 2026-09-25T15:37:17.561783Z
**Source:** seed_queries.jsonl (25 human-written seeds S036–S060)

## Audit Rules Applied
1. **Canonical location** in structured_query even when query uses colloquial spelling
2. **Correct start/end dates** for explicit, relative, festival, seasonal expressions where resolvable
3. **Correct sensor mapping** per production rules
4. **Correct object/event mapping**
4. **change_flag=true** whenever query explicitly asks about change, expansion, shift, damage, comparison over time
5. **Correct task_type** per SYSTEM_PROMPT rules
6. **Correct question/referring_expression** fields per production parser rules
7. **used_fallback=false, confidence≥0.9** for normal semantic gold labels (adversarial seeds keep fallback output)

## Critical Distinction
- **S045–S048 (adversarial)**: StructuredQuery MUST remain actual deterministic fallback output — DO NOT CHANGE
- **S036–S044, S049–S060 (normal)**: Gold label = semantic meaning per schema/system-prompt, NOT fallback output

---
## S036 [colloquial_placename] 🟢 NORMAL
**Query:** Show me Sentinel-2 images of agri land near Ghy from June 2024

**Current Label:**
- task_type: search
- location: Ghy
- start_date: 2024-06-01T00:00:00
- end_date: 2024-06-30T00:00:00
- sensor: Sentinel-2
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Location 'Ghy' is colloquial; should be canonical 'Guwahati'
- 'agri land'/'agriculture' → object should be 'agriculture'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Guwahati
- start_date: 2024-06-01T00:00:00
- end_date: 2024-06-30T00:00:00
- sensor: Sentinel-2
- object: agriculture
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S037 [colloquial_placename] 🟢 NORMAL
**Query:** What changed in Koli between 2023 and 2024 using SAR?

**Current Label:**
- task_type: change
- location: Koli
- start_date: None
- end_date: None
- sensor: Sentinel-1
- object: None
- event: None
- change_flag: True
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Location 'Koli' is colloquial; should be canonical 'Kolkata'
- Question words present → task_type should be 'vqa', not 'change'
- VQA query with '?' → question should be the full query text
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: vqa
- location: Kolkata
- start_date: None
- end_date: None
- sensor: Sentinel-1
- object: None
- event: None
- change_flag: True
- question: What changed in Koli between 2023 and 2024 using SAR?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S038 [colloquial_placename] 🟢 NORMAL
**Query:** Describe the land cover around Bambai coastal area

**Current Label:**
- task_type: caption
- location: Bambai
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Location 'Bambai' is colloquial; should be canonical 'Mumbai'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: caption
- location: Mumbai
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S039 [festival_seasonal_date] 🟢 NORMAL
**Query:** Show me Sentinel-2 images of agriculture near Guwahati after Rongali Bihu 2024

**Current Label:**
- task_type: search
- location: Guwahati
- start_date: None
- end_date: None
- sensor: Sentinel-2
- object: agriculture
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- 'after Rongali Bihu 2024' → late April 2024 (Rongali Bihu = Apr 14-20)
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Guwahati
- start_date: 2024-04-21T00:00:00
- end_date: 2024-04-30T00:00:00
- sensor: Sentinel-2
- object: agriculture
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S040 [festival_seasonal_date] 🟢 NORMAL
**Query:** What changed in Assam during Durga Puja season 2023?

**Current Label:**
- task_type: change
- location: Assam
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: True
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- 'Durga Puja season 2023' → October 2023 (Durga Puja = Oct 20-24)
- Question words present → task_type should be 'vqa', not 'change'
- VQA query with '?' → question should be the full query text
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: vqa
- location: Assam
- start_date: 2023-10-20T00:00:00
- end_date: 2023-10-24T00:00:00
- sensor: None
- object: None
- event: None
- change_flag: True
- question: What changed in Assam during Durga Puja season 2023?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S041 [sensor_phrasing] 🟢 NORMAL
**Query:** Show me cloud-penetrating radar images of Mumbai from January 2024

**Current Label:**
- task_type: search
- location: Mumbai
- start_date: 2024-01-01T00:00:00
- end_date: 2024-01-31T00:00:00
- sensor: Sentinel-1
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Mumbai
- start_date: 2024-01-01T00:00:00
- end_date: 2024-01-31T00:00:00
- sensor: Sentinel-1
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S042 [sensor_phrasing] 🟢 NORMAL
**Query:** Show me VV VH polarization data for Gujarat coast

**Current Label:**
- task_type: search
- location: Gujarat
- start_date: None
- end_date: None
- sensor: Sentinel-1
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Gujarat
- start_date: None
- end_date: None
- sensor: Sentinel-1
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S043 [change_event_phrasing] 🟢 NORMAL
**Query:** How much has the river course shifted in Assam since last year?

**Current Label:**
- task_type: search
- location: Assam
- start_date: 2025-01-01T00:00:00
- end_date: 2025-12-31T00:00:00
- sensor: None
- object: river
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- 'since last year' from 2025 perspective = 2024, not 2025
- Change language present → task_type should be 'change', not 'search'
- Query asks about change/shift/expansion/damage → change_flag should be true
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: change
- location: Assam
- start_date: 2024-01-01T00:00:00
- end_date: 2024-12-31T00:00:00
- sensor: None
- object: river
- event: None
- change_flag: True
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S044 [change_event_phrasing] 🟢 NORMAL
**Query:** Show me urban expansion in Bangalore between 2022 and 2024

**Current Label:**
- task_type: search
- location: Bangalore
- start_date: None
- end_date: None
- sensor: None
- object: urban
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- 'between 2022 and 2024' → 2022-01-01 to 2024-12-31
- Change language present → task_type should be 'change', not 'search'
- Query asks about change/shift/expansion/damage → change_flag should be true
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: change
- location: Bangalore
- start_date: 2022-01-01T00:00:00
- end_date: 2024-12-31T00:00:00
- sensor: None
- object: urban
- event: None
- change_flag: True
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S045 [adversarial_fallback] 🔴 ADVERSARIAL
**Query:** Show me Sentinel-2 images of agriculture near Dhaka from June 2024

**Current Label:**
- task_type: search
- location: Dhaka
- start_date: 2024-06-01T00:00:00
- end_date: 2024-06-30T00:00:00
- sensor: Sentinel-2
- object: agriculture
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Adversarial Seed — KEEP FALLBACK OUTPUT**
Per plan §4.3, adversarial seeds MUST use the actual fallback output as label.
No semantic correction applied.

**Proposed Corrected Gold Label:** SAME AS CURRENT (fallback output)

**Justification:** Adversarial seed; fallback output is the correct label per §4.3.

---
## S046 [adversarial_fallback] 🔴 ADVERSARIAL
**Query:** Show me flood extent before Bihu with no year specified

**Current Label:**
- task_type: search
- location: Bihu
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Adversarial Seed — KEEP FALLBACK OUTPUT**
Per plan §4.3, adversarial seeds MUST use the actual fallback output as label.
No semantic correction applied.

**Proposed Corrected Gold Label:** SAME AS CURRENT (fallback output)

**Justification:** Adversarial seed; fallback output is the correct label per §4.3.

---
## S047 [adversarial_fallback] 🔴 ADVERSARIAL
**Query:** Book me a flight to Guwahati for flood assessment

**Current Label:**
- task_type: search
- location: Guwahati
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Adversarial Seed — KEEP FALLBACK OUTPUT**
Per plan §4.3, adversarial seeds MUST use the actual fallback output as label.
No semantic correction applied.

**Proposed Corrected Gold Label:** SAME AS CURRENT (fallback output)

**Justification:** Adversarial seed; fallback output is the correct label per §4.3.

---
## S048 [adversarial_fallback] 🔴 ADVERSARIAL
**Query:** Show me flood in Assam and drought in Rajasthan at the same time

**Current Label:**
- task_type: search
- location: Assam
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Adversarial Seed — KEEP FALLBACK OUTPUT**
Per plan §4.3, adversarial seeds MUST use the actual fallback output as label.
No semantic correction applied.

**Proposed Corrected Gold Label:** SAME AS CURRENT (fallback output)

**Justification:** Adversarial seed; fallback output is the correct label per §4.3.

---
## S049 [disaster_officer] 🟢 NORMAL
**Query:** Is it safe to access Majuli island right now, is the embankment broken?

**Current Label:**
- task_type: search
- location: Majuli
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Question words present → task_type should be 'vqa', not 'search'
- 'breached embankment'/'embankment broken' → event should be 'breach'
- 'embankment' → object should be 'embankment'
- VQA query with '?' → question should be the full query text
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: vqa
- location: Majuli
- start_date: None
- end_date: None
- sensor: None
- object: embankment
- event: breach
- change_flag: False
- question: Is it safe to access Majuli island right now, is the embankment broken?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S050 [disaster_officer] 🟢 NORMAL
**Query:** Show me waterlogging extent in Guwahati as of today

**Current Label:**
- task_type: search
- location: Guwahati
- start_date: None
- end_date: None
- sensor: None
- object: water
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Guwahati
- start_date: None
- end_date: None
- sensor: None
- object: water
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S051 [disaster_officer] 🟢 NORMAL
**Query:** What is the flood situation in Kaziranga national park as of now?

**Current Label:**
- task_type: vqa
- location: Kaziranga
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: What is the flood situation in Kaziranga national park as of now?
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: vqa
- location: Kaziranga
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: What is the flood situation in Kaziranga national park as of now?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S052 [disaster_officer] 🟢 NORMAL
**Query:** Can we reach the flooded villages in Lakhimpur district today?

**Current Label:**
- task_type: search
- location: Lakhimpur
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Question words present → task_type should be 'vqa', not 'search'
- 'flood' in query → event should be 'flood' or 'flooding'
- VQA query with '?' → question should be the full query text
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: vqa
- location: Lakhimpur
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: flooding
- change_flag: False
- question: Can we reach the flooded villages in Lakhimpur district today?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S053 [disaster_officer] 🟢 NORMAL
**Query:** Show me breached embankments along Brahmaputra river as of today

**Current Label:**
- task_type: search
- location: Brahmaputra
- start_date: None
- end_date: None
- sensor: None
- object: river
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Location 'Brahmaputra' is colloquial; should be canonical 'Brahmaputra River'
- Change language present → task_type should be 'change', not 'search'
- Query asks about change/shift/expansion/damage → change_flag should be true
- 'breached embankment'/'embankment broken' → event should be 'breach'
- 'embankment' → object should be 'embankment'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: change
- location: Brahmaputra River
- start_date: None
- end_date: None
- sensor: None
- object: embankment
- event: breach
- change_flag: True
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S054 [disaster_officer] 🟢 NORMAL
**Query:** Is the road to Dibrugarh passable right now given the flooding?

**Current Label:**
- task_type: search
- location: Dibrugarh
- start_date: None
- end_date: None
- sensor: None
- object: road
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- 'flood' in query → event should be 'flood' or 'flooding'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Dibrugarh
- start_date: None
- end_date: None
- sensor: None
- object: road
- event: flooding
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S055 [disaster_officer] 🟢 NORMAL
**Query:** Show me inundation depth in Barpeta district for rescue planning

**Current Label:**
- task_type: search
- location: Barpeta
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- 'inundation' → object should be 'water'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Barpeta
- start_date: None
- end_date: None
- sensor: None
- object: water
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S056 [disaster_officer] 🟢 NORMAL
**Query:** What is the current flood level at Nimatighat gauge station?

**Current Label:**
- task_type: vqa
- location: Nimatighat
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: What is the current flood level at Nimatighat gauge station?
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Location 'Nimatighat' is colloquial; should be canonical 'Nimatighat Gauge Station'
- 'flood' in query → event should be 'flood' or 'flooding'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: vqa
- location: Nimatighat Gauge Station
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: flooding
- change_flag: False
- question: What is the current flood level at Nimatighat gauge station?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S057 [disaster_officer] 🟢 NORMAL
**Query:** Show me shelter locations accessible from flooded Golaghat area

**Current Label:**
- task_type: search
- location: Golaghat
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- 'flood' in query → event should be 'flood' or 'flooding'
- 'shelter locations accessible from...' → referring_expression should capture the grounding phrase
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: search
- location: Golaghat
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: flooding
- change_flag: False
- question: None
- referring_expression: shelter locations accessible from flooded area
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S058 [disaster_officer] 🟢 NORMAL
**Query:** Are the relief camps in Sonitpur district operational as of today?

**Current Label:**
- task_type: search
- location: Sonitpur
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Question words present → task_type should be 'vqa', not 'search'
- VQA query with '?' → question should be the full query text
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: vqa
- location: Sonitpur
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: Are the relief camps in Sonitpur district operational as of today?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S059 [disaster_officer] 🟢 NORMAL
**Query:** Show me erosion hotspots along Brahmaputra after last monsoon

**Current Label:**
- task_type: search
- location: Brahmaputra
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: None
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Location 'Brahmaputra' is colloquial; should be canonical 'Brahmaputra River'
- 'after last monsoon' → post-monsoon = October 2024
- Change language present → task_type should be 'change', not 'search'
- Query asks about change/shift/expansion/damage → change_flag should be true
- 'erosion' → object should be 'erosion'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: change
- location: Brahmaputra River
- start_date: 2024-10-01T00:00:00
- end_date: 2024-12-31T00:00:00
- sensor: None
- object: erosion
- event: None
- change_flag: True
- question: None
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.

---
## S060 [disaster_officer] 🟢 NORMAL
**Query:** Which bridges in Majuli are damaged by current floods?

**Current Label:**
- task_type: vqa
- location: Majuli
- start_date: None
- end_date: None
- sensor: None
- object: None
- event: None
- change_flag: False
- question: Which bridges in Majuli are damaged by current floods?
- referring_expression: None
- used_fallback: True
- confidence: 0.5

**Issues Found:**
- Change language present → task_type should be 'change', not 'vqa'
- Query asks about change/shift/expansion/damage → change_flag should be true
- 'flood' in query → event should be 'flood' or 'flooding'
- 'bridges' → object should be 'bridge'
- Normal training seed should have used_fallback=false (fallback is for adversarial only)
- Normal training seed confidence should be ≥0.9 (clear mapping), not 0.5 (fallback default)

**Proposed Corrected Gold Label:**
- task_type: change
- location: Majuli
- start_date: None
- end_date: None
- sensor: None
- object: bridge
- event: flooding
- change_flag: True
- question: Which bridges in Majuli are damaged by current floods?
- referring_expression: None
- used_fallback: False
- confidence: 0.95

**Justification:**
Normal training seed; gold label should reflect semantic meaning per schema/system-prompt, not fallback output.
