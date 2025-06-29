"""Utility functions: metric computation & logging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import evaluate

f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    return {"f1": f1, "accuracy": acc}


def save_label_map(label2id: Dict[str, int], out_dir: Path):
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)