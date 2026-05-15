"""
Models module for Policy Evaluate framework.

This module provides the model plugin system including:
- BaseModel: Abstract base class for all models
- ModelRegistry: Registry for discovering and loading models
- Model implementations: Qwen (HuggingFace and VLLM), Trained models, DynaGuard, GuardReasoner,
  Latent Policy Guard
"""

from .base import (
    BaseModel,
    ModelInfo,
    ModelOutput,
    ModelRegistry,
    register_model,
)

from .qwen import (
    QwenHuggingFaceModel,
    QwenVLLMModel,
    TrainedQwenModel,
)
from .guardreasoner import (
    GuardReasonerVLLMModel,
    GuardReasonerHFModel,
)
from .dynaguard import (
    DynaGuardHuggingface,
    parse_dynaguard_output,
    _extract_rule_from_explanation,
    _extract_rule_from_think,
)
from .latent_policy_guard import (
    LatentPolicyGuardModel,
)

__all__ = [
    "BaseModel",
    "ModelInfo",
    "ModelOutput",
    "ModelRegistry",
    "register_model",
    "QwenHuggingFaceModel",
    "QwenVLLMModel",
    "TrainedQwenModel",
    "GuardReasonerVLLMModel",
    "GuardReasonerHFModel",
    "DynaGuardHuggingface",
    "parse_dynaguard_output",
    "_extract_rule_from_explanation",
    "_extract_rule_from_think",
    "LatentPolicyGuardModel",
]
