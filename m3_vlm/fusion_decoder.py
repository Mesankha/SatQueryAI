"""
Fusion Decoder - Combines CROMA SAR tokens + EarthDial optical tokens → Shared Qwen decoder.
Implements cross-modal fusion for optical+SAR joint analysis.
"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.config import get_config

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module for combining optical and SAR tokens.
    
    Optical tokens (from EarthDial InternViT): (B, N_opt, D)
    SAR tokens (from CROMA + projector): (B, N_sar, D)
    
    Fusion: Cross-attention where optical attends to SAR and vice versa,
    then project each stream and concatenate.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_bidirectional = use_bidirectional
        
        # Cross-attention: optical queries attend to SAR keys/values
        self.optical_to_sar = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: SAR queries attend to optical keys/values
        if use_bidirectional:
            self.sar_to_optical = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        
        # Project each attended stream back to hidden_dim
        self.optical_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        
        self.sar_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        
        self._init_weights()
        logger.info(f"CrossAttentionFusion initialized: dim={hidden_dim}, heads={num_heads}, bidirectional={use_bidirectional}")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        optical_tokens: torch.Tensor,  # (B, N_opt, D)
        sar_tokens: torch.Tensor,      # (B, N_sar, D)
        optical_mask: Optional[torch.Tensor] = None,
        sar_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse optical and SAR tokens via cross-attention.
        
        Returns:
            Fused tokens: (B, N_opt + N_sar, D) - concatenated for decoder
        """
        # Optical attends to SAR
        optical_attended, _ = self.optical_to_sar(
            query=optical_tokens,
            key=sar_tokens,
            value=sar_tokens,
            key_padding_mask=sar_mask,
        )
        
        # SAR attends to optical (bidirectional)
        if self.use_bidirectional:
            sar_attended, _ = self.sar_to_optical(
                query=sar_tokens,
                key=optical_tokens,
                value=optical_tokens,
                key_padding_mask=optical_mask,
            )
        else:
            sar_attended = sar_tokens
        
        # Project each stream
        optical_projected = self.optical_proj(optical_attended)  # (B, N_opt, D)
        sar_projected = self.sar_proj(sar_attended)              # (B, N_sar, D)
        
        # Concatenate along sequence dimension
        fused = torch.cat([optical_projected, sar_projected], dim=1)  # (B, N_opt+N_sar, D)
        
        return fused


class ConcatProjectionFusion(nn.Module):
    """
    Simpler fusion: concatenate tokens + linear projection.
    Less compute, good baseline.
    """
    
    def __init__(self, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.fusion_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        optical_tokens: torch.Tensor,
        sar_tokens: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # Global pool each modality to single token, then concat
        # Or concatenate all patches (memory heavy)
        # Here: mean pool each -> (B, D) each -> concat -> project
        optical_pooled = optical_tokens.mean(dim=1)  # (B, D)
        sar_pooled = sar_tokens.mean(dim=1)          # (B, D)
        
        concat = torch.cat([optical_pooled, sar_pooled], dim=-1)  # (B, 2D)
        fused = self.fusion_proj(concat)  # (B, D)
        
        # Return as sequence of 1 token for decoder
        return fused.unsqueeze(1)  # (B, 1, D)


class FusionDecoder(nn.Module):
    """
    Complete fusion pipeline: dual encoder tokens → fusion → shared decoder.
    
    This module wraps the fusion logic and decoder generation.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        fusion_type: str = "cross_attention",  # "cross_attention" or "concat"
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        elif fusion_type == "concat":
            self.fusion = ConcatProjectionFusion(hidden_dim, dropout)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        logger.info(f"FusionDecoder initialized: fusion_type={fusion_type}")
    
    def forward(
        self,
        optical_tokens: torch.Tensor,
        sar_tokens: torch.Tensor,
        decoder_module: torch.nn.Module,
        tokenizer,
        max_new_tokens: int = 256,
        **generation_kwargs
    ) -> str | Tuple[str, torch.Tensor]:
        """
        Run full fusion + generation pipeline.
        
        Args:
            optical_tokens: (B, N_opt, D) from EarthDial InternViT
            sar_tokens: (B, N_sar, D) from CROMA + projector
            decoder_module: EarthDial's Qwen decoder
            tokenizer: EarthDial tokenizer
            max_new_tokens: Generation length
            
        Returns:
            Generated text string, and optionally fused embeddings
        """
        device = next(decoder_module.parameters()).device
        optical_tokens = optical_tokens.to(device)
        sar_tokens = sar_tokens.to(device)
        
        # Fuse tokens
        fused_tokens = self.fusion(optical_tokens, sar_tokens)  # (B, N_fused, D)
        
        # Create attention mask (all ones - no padding in fused tokens)
        attention_mask = torch.ones(
            fused_tokens.shape[:2], dtype=torch.long, device=device
        )
        
        # Generate using decoder
        with torch.no_grad():
            output_ids = decoder_module.generate(
                inputs_embeds=fused_tokens,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        # Decode
        input_len = fused_tokens.shape[1]
        new_tokens = output_ids[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip(), fused_tokens


def create_fusion_decoder(
    hidden_dim: int = 1024,
    fusion_type: str = "cross_attention",
    **kwargs
) -> FusionDecoder:
    """Factory function to create fusion decoder."""
    return FusionDecoder(hidden_dim, fusion_type, **kwargs)


# =============================================================================
# HIGH-LEVEL FUSION TASKS (caption, VQA) - for optical+SAR pairs
# =============================================================================

def run_fusion_caption(
    optical_tokens: torch.Tensor,
    sar_tokens: torch.Tensor,
    decoder_module: torch.nn.Module,
    tokenizer,
    max_new_tokens: int = 256
) -> str:
    """Run captioning on fused optical+SAR tokens."""
    fusion_decoder = create_fusion_decoder()
    caption, _ = fusion_decoder.forward(
        optical_tokens=optical_tokens,
        sar_tokens=sar_tokens,
        decoder_module=decoder_module,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    return caption


def run_fusion_vqa(
    optical_tokens: torch.Tensor,
    sar_tokens: torch.Tensor,
    question: str,
    decoder_module: torch.nn.Module,
    tokenizer,
    max_new_tokens: int = 256
) -> str:
    """Run VQA on fused optical+SAR tokens with question."""
    # Prepend question to fused tokens as text embedding
    # For simplicity, we'll use a prompt template approach
    # In practice, you'd embed the question and prepend to fused tokens
    
    fusion_decoder = create_fusion_decoder()
    
    # Embed question and prepend (simplified - just use text prompt to decoder)
    # Better approach: encode question with tokenizer, get embeddings, prepend
    question_ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids
    question_embeds = decoder_module.get_input_embeddings()(question_ids.to(decoder_module.device))
    
    # Fuse modalities first
    fused_tokens = fusion_decoder.fusion(optical_tokens, sar_tokens)
    
    # Prepend question embeddings
    combined_embeds = torch.cat([question_embeds, fused_tokens], dim=1)
    
    attention_mask = torch.ones(
        combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device
    )
    
    with torch.no_grad():
        output_ids = decoder_module.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    input_len = combined_embeds.shape[1]
    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()