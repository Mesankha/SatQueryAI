"""
M3 SAR Prompts and Templates — Specialized for Sentinel-1 SAR imagery.

These prompts are designed for fine-tuned GeoChat that needs to understand
SAR-specific visual semantics:
  - Bright/rough surfaces (urban, bare soil)
  - Dark/smooth surfaces (water, calm surfaces)
  - Volume scattering (vegetation, forests)
  - Layover, foreshortening, shadows
  - Speckle noise patterns

The prompts are intentionally verbose because the base GeoChat was never
exposed to SAR during training. The model needs explicit context to
interpret radar backscatter correctly.
"""

# ---------------------------------------------------------------------------
# System prompt for SAR understanding
# ---------------------------------------------------------------------------

# Base GeoChat system prompt (LLaVA-1.5 format)
GEOCHAT_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# SAR-specific system prompt - tells the model what it's looking at
SAR_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is an expert in remote sensing, with specialized knowledge "
    "of Synthetic Aperture Radar (SAR) imagery. SAR measures radar backscatter "
    "from the Earth's surface. Bright areas indicate rough surfaces or strong "
    "reflectors (urban, bare rock, metal). Dark areas indicate smooth surfaces "
    "(water, calm fields). Vegetation appears as moderate brightness with "
    "characteristic texture. The assistant interprets SAR features carefully "
    "and distinguishes them from optical appearances."
)


# ---------------------------------------------------------------------------
# Optical prompts (unchanged from before)
# ---------------------------------------------------------------------------

VQA_TEMPLATE = (
    f"{GEOCHAT_SYSTEM} "
    "USER: <image>\n{question} ASSISTANT:"
)

CAPTION_TEMPLATE = (
    f"{GEOCHAT_SYSTEM} "
    "USER: <image>\nPlease describe this remote sensing image in detail, "
    "including land cover, notable features, and any visible structures. ASSISTANT:"
)

GROUNDING_TEMPLATE = (
    f"{GEOCHAT_SYSTEM} "
    "USER: <image>\nLocate the following in the image and provide its bounding box "
    "as [x1, y1, x2, y2] in normalized coordinates (0-1): {expression} ASSISTANT:"
)


# ---------------------------------------------------------------------------
# SAR-specific prompts (NEW)
# ---------------------------------------------------------------------------

# These prompts include:
#  - Explicit SAR context (so model knows it's SAR, not optical)
#  - Backscatter interpretation guidance
#  - Polarization info (VV/VH)
#  - Tasks specific to SAR analysis

SAR_VQA_TEMPLATE = (
    f"{SAR_SYSTEM} "
    "USER: <image>\n"
    "This is a Sentinel-1 Synthetic Aperture Radar (SAR) image (C-band, "
    "VV and VH polarizations shown as RGB). The image shows radar "
    "backscatter intensity, NOT visible light. "
    "Bright tones = strong backscatter (urban, bare soil, metal structures). "
    "Dark tones = low backscatter (water, calm surfaces). "
    "Medium tones with texture = vegetation and forests. "
    "Answer the following question based on radar backscatter: {question} "
    "ASSISTANT:"
)

SAR_CAPTION_TEMPLATE = (
    f"{SAR_SYSTEM} "
    "USER: <image>\n"
    "This is a Sentinel-1 Synthetic Aperture Radar (SAR) image (C-band, "
    "VV and VH polarizations shown as RGB). Describe what you observe in "
    "terms of:\n"
    "1. Backscatter intensity (bright/dark regions and their meaning)\n"
    "2. Land cover types (urban, water, vegetation, bare soil, agriculture)\n"
    "3. Spatial patterns and textures\n"
    "4. Notable features (linear structures like roads, geometric patterns "
    "of fields, irregular shapes of natural features)\n"
    "5. Possible interpretation of the scene\n"
    "ASSISTANT:"
)

SAR_GROUNDING_TEMPLATE = (
    f"{SAR_SYSTEM} "
    "USER: <image>\n"
    "This is a Sentinel-1 SAR image. Locate the following in the image "
    "and provide its bounding box as [x1, y1, x2, y2] in normalized "
    "coordinates (0-1). Look for the feature based on its expected "
    "backscatter characteristics: {expression} ASSISTANT:"
)


# ---------------------------------------------------------------------------
# Multi-modal prompts (optical + SAR paired) - NEW
# ---------------------------------------------------------------------------

# For fusion: when user has both optical and SAR of same scene
FUSION_CAPTION_TEMPLATE = (
    f"{GEOCHAT_SYSTEM} "
    "USER: <image>\nThis image shows the same area captured by both Sentinel-2 "
    "optical (RGB) and Sentinel-1 SAR (VV+VH+ratio as RGB) satellites. "
    "Describe the scene by combining insights from both modalities:\n"
    "- From optical: land cover types, colors, vegetation health, water clarity\n"
    "- From SAR: surface roughness, structural features, moisture content\n"
    "- Together: how does the SAR signal complement the optical view?\n"
    "ASSISTANT:"
)

FUSION_VQA_TEMPLATE = (
    f"{GEOCHAT_SYSTEM} "
    "USER: <image>\n"
    "This image combines Sentinel-2 optical and Sentinel-1 SAR views of "
    "the same area. Use BOTH modalities to answer: {question} "
    "ASSISTANT:"
)

FUSION_CHANGE_TEMPLATE = (
    f"{GEOCHAT_SYSTEM} "
    "USER: <image>\n"
    "These are bi-temporal Sentinel-1 SAR images of the same area, showing "
    "the scene before and after some event. Describe what changed. "
    "Focus on:\n"
    "- Changes in backscatter intensity (brighter/darker)\n"
    "- Changes in spatial patterns (new structures, removed features)\n"
    "- Likely cause of changes (flooding, urban growth, deforestation, "
    "agricultural changes)\n"
    "ASSISTANT:"
)


# ---------------------------------------------------------------------------
# Modality-aware prompt selector
# ---------------------------------------------------------------------------

def select_prompt(
    task: str,
    modality: str,
    **kwargs,
) -> str:
    """
    Select the appropriate prompt based on task and modality.

    Args:
        task: "vqa", "caption", "grounding", "change", "fusion"
        modality: "optical", "sar", "fusion"
        **kwargs: format arguments (question, expression, etc.)

    Returns:
        Formatted prompt string

    Example:
        >>> prompt = select_prompt("caption", "sar")
        >>> prompt = select_prompt("vqa", "fusion", question="Is there flooding?")
    """
    if modality == "sar":
        templates = {
            "vqa": SAR_VQA_TEMPLATE,
            "caption": SAR_CAPTION_TEMPLATE,
            "grounding": SAR_GROUNDING_TEMPLATE,
            "change": SAR_CAPTION_TEMPLATE,  # SAR change uses caption template
            "fusion": SAR_CAPTION_TEMPLATE,
        }
    elif modality == "fusion":
        templates = {
            "vqa": FUSION_VQA_TEMPLATE,
            "caption": FUSION_CAPTION_TEMPLATE,
            "grounding": GROUNDING_TEMPLATE,  # fallback to optical grounding
            "change": FUSION_CHANGE_TEMPLATE,
            "fusion": FUSION_CAPTION_TEMPLATE,
        }
    else:  # optical
        templates = {
            "vqa": VQA_TEMPLATE,
            "caption": CAPTION_TEMPLATE,
            "grounding": GROUNDING_TEMPLATE,
            "change": CAPTION_TEMPLATE,
            "fusion": FUSION_CAPTION_TEMPLATE,
        }

    template = templates.get(task, VQA_TEMPLATE)
    try:
        return template.format(**kwargs)
    except KeyError:
        # Missing required kwarg - return without formatting
        return template