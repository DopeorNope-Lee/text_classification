"""Unified model builder supporting Linear Probing + optional LoRA.

* If ``linear_probing`` → backbone parameters are frozen.
* If ``use_lora``        → LoRA adapters are injected **with target modules
  chosen dynamically** to match the underlying architecture (BERT vs GPT‑2).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    pretrained_model_name: str = field(default_factory=lambda: os.getenv("BASE_MODEL", "skt/kobert-base-v1"))
    num_labels: int = 3
    linear_probing: bool = False  # freeze backbone if True
    use_lora: bool = True         # attach LoRA adapters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

# ---------------------------------------------------------------------------
# Helper to pick LoRA targets
# ---------------------------------------------------------------------------

def _get_target_modules(model_type: str) -> List[str]:
    """Return list of module name substrings for LoRA injection."""
    if model_type in {"bert", "kobert"}:   # BERT family (incl. KoBERT)
        return ["query", "key", "value"]
    if model_type == "gpt2":                # GPT‑2 fused qkv proj layer
        return ["c_attn"]
    # Fallback – attempt generic names
    return ["q_proj", "v_proj"]

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig):
    config = AutoConfig.from_pretrained(cfg.pretrained_model_name, num_labels=cfg.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.pretrained_model_name, config=config)

    # 1) Linear probing (freeze backbone)
    if cfg.linear_probing:
        base_prefix = getattr(model, "base_model_prefix", None)
        if base_prefix and hasattr(model, base_prefix):
            backbone = getattr(model, base_prefix)
            backbone.requires_grad_(False)  # freeze all
        else:
            for n, p in model.named_parameters():
                if "classifier" not in n and "score" not in n:
                    p.requires_grad = False

    # 2) LoRA adapters
    if cfg.use_lora:
        targets = _get_target_modules(config.model_type)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)

    return model