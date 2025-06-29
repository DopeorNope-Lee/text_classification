from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on a single text")
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--num_labels", type=int, required=True)
    return p.parse_args()

def encode(text, tok):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    if "token_type_ids" in enc:           # BERT 류만 존재 → 0 으로 통일
        enc["token_type_ids"].zero_()
    return enc


def main():
    args = parse_args()
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    with torch.no_grad():
        inputs = encode(args.text, tok)
        logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()
    print("Predicted label:", pred)


if __name__ == "__main__":
    main()