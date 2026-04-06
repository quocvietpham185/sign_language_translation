"""
===============================================================
  UPGRADED HST-GNN FOR SIGN LANGUAGE TRANSLATION
  Based on: arxiv 2111.07258 — Kan et al., 2021
  Upgrades:
    1. MediaPipe keypoints thay ResNet-152 (không cần GPU preprocess)
    2. Lightweight Graph Encoder (GCN + Graph Transformer)
    3. mBART-50 pretrained decoder (thay 2×LSTM)
    4. CTC + Cross-Entropy joint loss
    5. Mixed Precision Training (AMP)
    6. Gradient Checkpointing (tiết kiệm VRAM)
    7. Multi-dataset support (PHOENIX, WLASL, CSL, Kaggle)
    8. Data Augmentation (flip, noise, time stretch, dropout)
    9. Cosine LR + Warmup scheduler
   10. Auto-resume từ checkpoint (Colab-friendly)
===============================================================
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import get_config
from dataset import SignLanguageDataset, collate_fn
from model import UpgradedHSTGNN
from utils import (
    set_seed, setup_logger, save_checkpoint, load_checkpoint,
    compute_wer, compute_bleu, AverageMeter, EarlyStopping
)
from scheduler import WarmupCosineScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Upgraded HST-GNN Training")
    parser.add_argument("--config", type=str, default="configs/phoenix.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Output directory")
    parser.add_argument("--drive_backup", type=str, default=None,
                        help="Google Drive path for checkpoint backup")
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scaler, scheduler,
                    criterion_ctc, criterion_ce, config, epoch, logger):
    model.train()
    losses = AverageMeter()
    ctc_losses = AverageMeter()
    ce_losses = AverageMeter()

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        keypoints = batch["keypoints"].to(config.device)        # (B, T, N, 3)
        gloss_ids = batch["gloss_ids"].to(config.device)        # (B, Lg)
        text_ids = batch["text_ids"].to(config.device)          # (B, Lw)
        gloss_lengths = batch["gloss_lengths"].to(config.device)
        text_lengths = batch["text_lengths"].to(config.device)
        keypoint_lengths = batch["keypoint_lengths"].to(config.device)

        with autocast(enabled=config.use_amp):
            outputs = model(
                keypoints=keypoints,
                keypoint_lengths=keypoint_lengths,
                gloss_targets=gloss_ids,
                text_targets=text_ids,
            )

            # CTC loss (feats → gloss)
            log_probs = outputs["gloss_log_probs"]              # (T, B, vocab)
            input_lengths = outputs["encoder_lengths"]          # (B,)
            loss_ctc = criterion_ctc(
                log_probs, gloss_ids,
                input_lengths, gloss_lengths
            )

            # Cross-Entropy loss (gloss → text via mBART decoder)
            logits = outputs["text_logits"]                     # (B, Lw, vocab)
            B, Lw, V = logits.shape
            loss_ce = criterion_ce(
                logits.reshape(B * Lw, V),
                text_ids.reshape(B * Lw)
            )

            loss = (config.lambda_ctc * loss_ctc +
                    config.lambda_ce * loss_ce)
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        B = keypoints.size(0)
        losses.update(loss.item() * config.gradient_accumulation_steps, B)
        ctc_losses.update(loss_ctc.item(), B)
        ce_losses.update(loss_ce.item(), B)

        if step % config.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(dataloader)}] "
                f"Loss: {losses.avg:.4f} CTC: {ctc_losses.avg:.4f} "
                f"CE: {ce_losses.avg:.4f} LR: {lr:.2e}"
            )

    return {"loss": losses.avg, "ctc_loss": ctc_losses.avg, "ce_loss": ce_losses.avg}


@torch.no_grad()
def evaluate(model, dataloader, gloss_vocab, text_vocab, config, logger, split="val"):
    model.eval()
    all_gloss_preds, all_gloss_gts = [], []
    all_text_preds, all_text_gts = [], []
    losses = AverageMeter()

    criterion_ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=text_vocab.pad_id)

    for batch in dataloader:
        keypoints = batch["keypoints"].to(config.device)
        gloss_ids = batch["gloss_ids"].to(config.device)
        text_ids = batch["text_ids"].to(config.device)
        gloss_lengths = batch["gloss_lengths"].to(config.device)
        keypoint_lengths = batch["keypoint_lengths"].to(config.device)

        with autocast(enabled=config.use_amp):
            outputs = model(
                keypoints=keypoints,
                keypoint_lengths=keypoint_lengths,
                gloss_targets=gloss_ids,
                text_targets=text_ids,
            )

        # Decode predictions
        gloss_preds = model.decode_gloss(outputs["gloss_log_probs"])
        text_preds = model.decode_text(
            outputs["encoder_hidden"],
            outputs["encoder_lengths"],
            text_vocab,
            config.max_text_len
        )

        # Convert IDs to strings
        for i in range(keypoints.size(0)):
            gp = gloss_vocab.ids_to_text(gloss_preds[i])
            gg = gloss_vocab.ids_to_text(gloss_ids[i].cpu().tolist())
            all_gloss_preds.append(gp)
            all_gloss_gts.append(gg)

            tp = text_vocab.ids_to_text(text_preds[i])
            tg = text_vocab.ids_to_text(text_ids[i].cpu().tolist())
            all_text_preds.append(tp)
            all_text_gts.append(tg)

    wer = compute_wer(all_gloss_preds, all_gloss_gts)
    bleu1, bleu2, bleu3, bleu4 = compute_bleu(all_text_preds, all_text_gts)

    logger.info(
        f"[{split.upper()}] WER: {wer:.2f}% | "
        f"BLEU-1: {bleu1:.2f} BLEU-2: {bleu2:.2f} "
        f"BLEU-3: {bleu3:.2f} BLEU-4: {bleu4:.2f}"
    )

    return {
        "wer": wer,
        "bleu1": bleu1, "bleu2": bleu2,
        "bleu3": bleu3, "bleu4": bleu4,
    }


def main():
    args = parse_args()
    config = get_config(args.config)
    config.output_dir = args.output_dir
    config.drive_backup = args.drive_backup

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config.output_dir)
    set_seed(config.seed)

    logger.info("=" * 60)
    logger.info("  UPGRADED HST-GNN TRAINING")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Device: {config.device}")
    logger.info("=" * 60)

    # ── Datasets ──────────────────────────────────────────────
    from vocabulary import Vocabulary
    gloss_vocab = Vocabulary.load(config.gloss_vocab_path)
    text_vocab = Vocabulary.load(config.text_vocab_path)

    train_dataset = SignLanguageDataset(
        data_path=config.train_data_path,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        config=config,
        split="train",
        augment=True,
    )
    val_dataset = SignLanguageDataset(
        data_path=config.val_data_path,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        config=config,
        split="val",
        augment=False,
    )
    test_dataset = SignLanguageDataset(
        data_path=config.test_data_path,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        config=config,
        split="test",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.eval_batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.eval_batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────
    model = UpgradedHSTGNN(
        num_keypoints=config.num_keypoints,
        keypoint_dim=config.keypoint_dim,
        d_model=config.d_model,
        num_graph_layers=config.num_graph_layers,
        num_heads=config.num_heads,
        num_gloss_classes=len(gloss_vocab),
        text_vocab_size=len(text_vocab),
        decoder_name=config.decoder_name,
        dropout=config.dropout,
        use_gradient_checkpointing=config.gradient_checkpointing,
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params/1e6:.2f}M | Trainable: {train_params/1e6:.2f}M")

    # ── Loss ──────────────────────────────────────────────────
    criterion_ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=text_vocab.pad_id)

    # ── Optimizer ─────────────────────────────────────────────
    # Phân biệt pretrained (LR nhỏ) và scratch (LR lớn)
    pretrained_params = []
    scratch_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_decoder" in name:
            pretrained_params.append(param)
        else:
            scratch_params.append(param)

    optimizer = optim.AdamW([
        {"params": scratch_params, "lr": config.lr},
        {"params": pretrained_params, "lr": config.lr * config.pretrained_lr_scale},
    ], weight_decay=config.weight_decay)

    # ── Scheduler ─────────────────────────────────────────────
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(patience=config.patience, mode="min")

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 1
    best_wer = float("inf")
    if args.resume:
        start_epoch, best_wer = load_checkpoint(args.resume, model, optimizer, scheduler, logger)
    else:
        # Auto-detect checkpoint trong output_dir
        ckpt_path = Path(config.output_dir) / "checkpoint_latest.pt"
        if ckpt_path.exists():
            start_epoch, best_wer = load_checkpoint(str(ckpt_path), model, optimizer, scheduler, logger)

    if args.eval_only:
        logger.info("Running evaluation only...")
        evaluate(model, test_loader, gloss_vocab, text_vocab, config, logger, "test")
        return

    # ── Training Loop ─────────────────────────────────────────
    logger.info(f"Starting training from epoch {start_epoch}...")
    history = []

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            criterion_ctc, criterion_ce, config, epoch, logger
        )

        val_metrics = evaluate(model, val_loader, gloss_vocab, text_vocab, config, logger, "val")

        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch} done in {elapsed:.1f}s")

        # Log history
        history.append({"epoch": epoch, **train_metrics, **val_metrics})
        with open(Path(config.output_dir) / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save checkpoints
        is_best = val_metrics["wer"] < best_wer
        if is_best:
            best_wer = val_metrics["wer"]
            logger.info(f"✓ New best WER: {best_wer:.2f}%")

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=val_metrics,
            best_wer=best_wer,
            output_dir=config.output_dir,
            is_best=is_best,
            drive_backup=config.drive_backup,
        )

        # Early stopping
        if early_stopping(val_metrics["wer"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Final Test ────────────────────────────────────────────
    logger.info("Loading best checkpoint for final test...")
    best_ckpt = Path(config.output_dir) / "checkpoint_best.pt"
    load_checkpoint(str(best_ckpt), model, None, None, logger)
    test_metrics = evaluate(model, test_loader, gloss_vocab, text_vocab, config, logger, "test")

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS:")
    logger.info(f"  WER:    {test_metrics['wer']:.2f}%")
    logger.info(f"  BLEU-1: {test_metrics['bleu1']:.2f}")
    logger.info(f"  BLEU-2: {test_metrics['bleu2']:.2f}")
    logger.info(f"  BLEU-3: {test_metrics['bleu3']:.2f}")
    logger.info(f"  BLEU-4: {test_metrics['bleu4']:.2f}")
    logger.info("=" * 60)

    with open(Path(config.output_dir) / "final_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
