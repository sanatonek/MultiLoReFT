"""Baseline model implementations with unified encode/get_components_for_eval interface."""
from .apollo import Apollo, Encoder, Decoder
from .contrastive import (
    CrossAttentionFusion,
    AttentionFusion,
    LinearHead,
    contrastive_loss,
    ContrastiveModel,
)
from .drim import DRIM_U, MLP_EncShared, MLP_EncUnique, MLP_DecUnique

__all__ = [
    "Apollo",
    "Encoder",
    "Decoder",
    "CrossAttentionFusion",
    "AttentionFusion",
    "LinearHead",
    "contrastive_loss",
    "ContrastiveModel",
    "DRIM_U",
    "MLP_EncShared",
    "MLP_EncUnique",
    "MLP_DecUnique",
]
