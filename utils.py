"""
utils.py — Training utilities
"""

import os
import random
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
#  SEED
# ══════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════

def setup_logger(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("HST-GNN")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_file = Path(output_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ══════════════════════════════════════════════════════════════
#  AVERAGE METER
# ══════════════════════════════════════════════════════════════

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════

def compute_wer(preds: list, refs: list) -> float:
    """
    Word Error Rate (%).
    Dùng dynamic programming — không cần thư viện ngoài.
    """
    total_errors = 0
    total_words = 0
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        total_words += len(ref_tokens)
        total_errors += _edit_distance(pred_tokens, ref_tokens)
    if total_words == 0:
        return 0.0
    return (total_errors / total_words) * 100.0


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two token lists."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        ndp = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                ndp[j] = dp[j - 1]
            else:
                ndp[j] = 1 + min(dp[j], ndp[j - 1], dp[j - 1])
        dp = ndp
    return dp[n]


def compute_bleu(preds: list, refs: list, max_n: int = 4):
    """
    BLEU-1 through BLEU-4 calculation.
    Corpus-level BLEU với brevity penalty.
    """
    from collections import Counter
    import math

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    total_pred_len = 0
    total_ref_len = 0

    for pred, ref in zip(preds, refs):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        total_pred_len += len(pred_tokens)
        total_ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ng = ngrams(pred_tokens, n)
            ref_ng = ngrams(ref_tokens, n)
            for gram, cnt in pred_ng.items():
                clipped_counts[n-1] += min(cnt, ref_ng.get(gram, 0))
            total_counts[n-1] += max(0, len(pred_tokens) - n + 1)

    bleu_scores = []
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            bleu_scores.append(0.0)
        else:
            bleu_scores.append(clipped_counts[n] / total_counts[n])

    # Brevity penalty
    if total_pred_len == 0:
        bp = 0.0
    elif total_pred_len >= total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_pred_len)

    bleus = []
    log_sum = 0.0
    for n in range(max_n):
        p = bleu_scores[n]
        if p > 0:
            log_sum += math.log(p)
        bleu_n = bp * math.exp(log_sum / (n + 1)) * 100.0 if p > 0 else 0.0
        bleus.append(round(bleu_n, 2))

    return tuple(bleus)  # (bleu1, bleu2, bleu3, bleu4)


# ══════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    metrics: dict,
    best_wer: float,
    output_dir: str,
    is_best: bool = False,
    drive_backup: Optional[str] = None,
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "best_wer": best_wer,
    }

    output_dir = Path(output_dir)
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(state, latest_path)

    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        shutil.copy(latest_path, best_path)
        print(f"✓ Saved best checkpoint (WER: {best_wer:.2f}%)")

    # Backup to Google Drive (Colab)
    if drive_backup:
        try:
            drive_dir = Path(drive_backup)
            drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(latest_path, drive_dir / "checkpoint_latest.pt")
            if is_best:
                shutil.copy(latest_path, drive_dir / "checkpoint_best.pt")
            print(f"✓ Backed up to Drive: {drive_backup}")
        except Exception as e:
            print(f"Warning: Drive backup failed: {e}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    logger=None,
):
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    log(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location="cpu")

    model.load_state_dict(state["model_state_dict"])

    if optimizer and state.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:
            log(f"Warning: optimizer state load failed: {e}")

    if scheduler and state.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            log(f"Warning: scheduler state load failed: {e}")

    epoch = state.get("epoch", 0) + 1
    best_wer = state.get("best_wer", float("inf"))
    metrics = state.get("metrics", {})
    log(f"Resumed from epoch {epoch-1}, best WER: {best_wer:.2f}%")
    log(f"Last metrics: {metrics}")

    return epoch, best_wer


# ══════════════════════════════════════════════════════════════
#  EARLY STOPPING
# ══════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def __call__(self, value: float) -> bool:
        improved = (
            value < self.best - self.min_delta if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
