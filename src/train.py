from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding

from src.data_loader import get_dataset
from src.model import ModelConfig, build_model



class SmartCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        # BERT류만 전체 0으로 세팅
        if "token_type_ids" in batch:
            batch["token_type_ids"].zero_()  
        return batch


def parse_args():
    p = argparse.ArgumentParser(description="Train customer‑complaints classifier")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--linear_probing", action="store_true")
    p.add_argument("--no_use_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) 토크나이저 단 한 번 생성
    model_name = args.model_name
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # 2) GPT-2일 때만 pad 토큰 추가
    added_pad = False
    if tokenizer.pad_token is None and "gpt2" in model_name.lower():
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        added_pad = True

    # 3) 같은 토크나이저로 데이터 전처리
    ds, label2id = get_dataset(tokenizer, args.max_len)


    # 4) 모델 빌드 → 토큰 추가됐으면 임베딩 크기 확장
    cfg   = ModelConfig(pretrained_model_name=model_name,
                        num_labels=len(label2id),
                        linear_probing=args.linear_probing,
                        use_lora=args.use_lora,
                        lora_r=args.lora_r)
    model = build_model(cfg)
    
    if added_pad:
        model.resize_token_embeddings(len(tokenizer))
    
    model.config.pad_token_id = tokenizer.pad_token_id 


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_steps=50,
        eval_steps=50,
    )

    data_collator = SmartCollator(tokenizer, return_tensors="pt")


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()