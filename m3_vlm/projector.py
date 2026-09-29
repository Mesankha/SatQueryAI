"""
CROMA Projector - Projects CROMA patch tokens to shared embedding space.
2-layer MLP: CROMA hidden_dim (768 for ViT-B) -> 1024 -> 1024 (shared space).
Shared space matches EarthDial InternViT output dimension for fusion.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from shared.config import get_config

logger = logging.getLogger(__name__)


class CROMAProjector(nn.Module):
    """
    Projector from CROMA encoder space to shared embedding space.
    
    Architecture: 2-layer MLP with GELU activation
    - Input: CROMA patch tokens (B, num_patches, croma_dim) 
    - Hidden: 1024
    - Output: Shared embedding space (B, num_patches, 1024)
    
    CROMA ViT-Base: 768 dim -> 1024 -> 1024
    CROMA ViT-Large: 1024 dim -> 1024 -> 1024 (identity-ish)
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # CROMA ViT-B hidden dim
        hidden_dim: int = 1024,
        output_dim: int = 1024,  # Shared space (matches EarthDial InternViT)
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"CROMAProjector initialized: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def _init_weights(self):
        """Initialize projector weights."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, croma_tokens: torch.Tensor) -> torch.Tensor:
        """
        Project CROMA tokens to shared space.
        
        Args:
            croma_tokens: (B, num_patches, croma_dim)
            
        Returns:
            projected: (B, num_patches, output_dim)
        """
        return self.mlp(croma_tokens)


def create_croma_projector(
    croma_variant: str = "base",
    shared_dim: int = 1024
) -> CROMAProjector:
    """
    Factory function to create projector matching CROMA variant.
    
    Args:
        croma_variant: "base" (768 dim) or "large" (1024 dim)
        shared_dim: Target shared embedding dimension (default 1024 for EarthDial)
        
    Returns:
        CROMAProjector instance
    """
    if croma_variant == "base":
        input_dim = 768
    elif croma_variant == "large":
        input_dim = 1024
    else:
        raise ValueError(f"Unknown CROMA variant: {croma_variant}")
    
    return CROMAProjector(
        input_dim=input_dim,
        hidden_dim=shared_dim,
        output_dim=shared_dim,
    )


def load_croma_projector(
    projector_path: str,
    croma_variant: str = "base",
    shared_dim: int = 1024,
    device: str = "cpu"
) -> CROMAProjector:
    """
    Load trained projector from checkpoint.
    
    Args:
        projector_path: Path to projector .pt file
        croma_variant: "base" or "large"
        shared_dim: Shared embedding dimension
        device: Device to load on
        
    Returns:
        Loaded CROMAProjector in eval mode
    """
    projector = create_croma_projector(croma_variant, shared_dim)
    
    try:
        state_dict = torch.load(projector_path, map_location=device)
        projector.load_state_dict(state_dict)
        projector.to(device)
        projector.eval()
        logger.info(f"Loaded CROMA projector from {projector_path}")
        return projector
    except Exception as e:
        logger.warning(f"Failed to load projector from {projector_path}: {e}")
        logger.info("Using randomly initialized projector")
        projector.to(device)
        projector.eval()
        return projector


def save_croma_projector(projector: CROMAProjector, path: str) -> None:
    """Save projector state dict."""
    torch.save(projector.state_dict(), path)
    logger.info(f"Saved CROMA projector to {path}")


# For inference without training - identity-like projector if dims match
class IdentityProjector(nn.Module):
    """Fallback projector when CROMA dim already matches shared space (e.g., ViT-Large)."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if input_dim == output_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(input_dim, output_dim, bias=False)
            nn.init.eye_(self.proj.weight) if input_dim == output_dim else nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def get_projector_for_inference(
    croma_variant: str = "base",
    shared_dim: int = 1024,
    projector_path: Optional[str] = None,
    device: str = "cpu"
) -> nn.Module:
    """
    Get projector for inference - loads trained or creates appropriate fallback.
    
    Priority:
    1. Load from projector_path if exists
    2. Create trained projector architecture (random init - for structure)
    3. Identity projector if dims match
    """
    if projector_path:
        try:
            return load_croma_projector(projector_path, croma_variant, shared_dim, device)
        except Exception as e:
            logger.warning(f"Could not load projector from {projector_path}: {e}")
    
    # Create architecture-appropriate projector
    if croma_variant == "large" and shared_dim == 1024:
        # ViT-Large already 1024-dim - use identity with optional linear
        return IdentityProjector(1024, shared_dim).to(device)
    else:
        # ViT-Base (768) -> 1024 needs projection
        return create_croma_projector(croma_variant, shared_dim).to(device)