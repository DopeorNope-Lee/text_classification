"""Post‑training dynamic quantization / 8‑bit with bitsandbytes."""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoConfig
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Path to fine‑tuned checkpoint")
    p.add_argument("--output", type=Path, default="./models/quantized")
    p.add_argument("--bnb_4bit", action="store_true", help="Use 4‑bit bitsandbytes quantization")
    return p.parse_args()


def main():
    args = parse_args()

    if args.bnb_4bit:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.ckpt,
            load_in_4bit=True,
            device_map="auto",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
        model.to(torch.float32)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    print(f"Quantized model saved to {args.output}")


if __name__ == "__main__":
    main()