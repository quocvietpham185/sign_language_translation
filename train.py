"""
===============================================================
  UPGRADED HST-GNN FOR SIGN LANGUAGE TRANSLATION v2
  Based on: arxiv 2111.07258 — Kan et al., 2021

  Upgrades v2:
    1. Fix CTC loss bug với isolated sign datasets (WLASL)
       → dataset_mode="continuous" mới dùng CTC
       → dataset_mode="isolated" chỉ dùng CE loss
    2. Sliding Window Transformer trong TemporalEncoder
    3. Timed Drive auto-save (mỗi N phút, không cần thủ công)
    4. 2-phase training:
       Phase 1 → Pretrain graph encoder + CTC (freeze mBART decoder)
       Phase 2 → Fine-tune decoder (freeze encoder)
    5. Per-step checkpoint backup (đảm bảo không mất data khi Colab die)
===============================================================
"""

import os
import sys
import json
import time
import shutil
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
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
    parser = argparse.ArgumentParser(description="Upgraded HST-GNN v2 Training")
    parser.add_argument("--config", type=str, default="configs/colab.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--phase", type=int, default=0,
                        help="0=joint, 1=pretrain encoder only, 2=finetune decoder only")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--drive_backup", type=str, default=None,
                        help="Google Drive path for timed checkpoint backup")
    parser.add_argument("--save_interval_min", type=int, default=15,
                        help="Auto-save to Drive every N minutes (Colab session safety)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
#  COLAB AUTO-SAVER — Lưu Drive theo thời gian thực
#  Nếu Colab die sau 3h, checkpoint gần nhất trên Drive sẽ được
#  dùng để resume, không mất quá N phút training.
# ══════════════════════════════════════════════════════════════

class TimedDriveSaver:
    """
    Auto-save checkpoint lên Google Drive mỗi N phút.
    Không phụ thuộc vào epoch boundary — an toàn với session giới hạn.
    """
    def __init__(self, drive_path: str, interval_min: int = 15):
        self.drive_path = Path(drive_path) if drive_path else None
        self.interval_sec = interval_min * 60
        self._last_save = time.time()

    def should_save(self) -> bool:
        return (self.drive_path is not None and
                time.time() - self._last_save >= self.interval_sec)

    def save(self, local_path: str, tag: str = "timed"):
        if self.drive_path is None:
            return
        try:
            self.drive_path.mkdir(parents=True, exist_ok=True)
            dest = self.drive_path / f"checkpoint_{tag}.pt"
            shutil.copy(local_path, dest)
            self._last_save = time.time()
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [Drive] ✓ Saved {tag} checkpoint at {ts} → {dest}")
        except Exception as e:
            print(f"  [Drive] ✗ Backup failed: {e}")

    def force_save(self, local_path: str, tag: str = "latest"):
        """Gọi cuối epoch bất kể timer."""
        self._last_save = 0  # reset để trigger ngay
        self.save(local_path, tag)


# ══════════════════════════════════════════════════════════════
#  PHASE CONTROL — Đóng/mở phần model theo training phase
# ══════════════════════════════════════════════════════════════

def set_training_phase(model: UpgradedHSTGNN, phase: int, logger):
    """
    Phase 0 (joint): Train tất cả (default)
    Phase 1 (pretrain): Freeze mBART decoder → chỉ train graph/temporal encoder
    Phase 2 (finetune): Freeze encoder → chỉ train mBART decoder

    Chiến lược 2-phase giúp:
    - Phase 1 trên Colab free (2-3h): Encoder học keypoint → gloss tốt
    - Phase 2 trên Colab free (2-3h): Decoder học gloss → text
    """
    if phase == 0:
        for p in model.parameters():
            p.requires_grad = True
        logger.info("Phase 0: Training ALL parameters jointly")

    elif phase == 1:
        # Freeze mBART decoder
        for p in model.text_decoder.parameters():
            p.requires_grad = False
        # Unfreeze encoder
        for p in model.graph_encoder.parameters():
            p.requires_grad = True
        for p in model.temporal_encoder.parameters():
            p.requires_grad = True
        for p in model.gloss_head.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 1 (Pretrain Encoder): {trainable/1e6:.1f}M trainable params "
                    f"(mBART frozen)")

    elif phase == 2:
        # Freeze encoder
        for p in model.graph_encoder.parameters():
            p.requires_grad = False
        for p in model.temporal_encoder.parameters():
            p.requires_grad = False
        # Unfreeze mBART (but respect its own internal freeze)
        for name, p in model.text_decoder.named_parameters():
            if not ("shared" in name or "embed" in name or
                    any(f"decoder.layers.{i}" in name for i in range(6))):
                p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 2 (Finetune Decoder): {trainable/1e6:.1f}M trainable params "
                    f"(encoder frozen)")


# ══════════════════════════════════════════════════════════════
#  LOSS ROUTING — dataset_mode aware
# ══════════════════════════════════════════════════════════════

def compute_loss(
    outputs: dict,
    batch: dict,
    criterion_ctc: nn.CTCLoss,
    criterion_ce: nn.CrossEntropyLoss,
    config,
) -> tuple:
    """
    FIX: CTC loss chỉ áp dụng cho "continuous" samples (PHOENIX, CSL).
    "isolated" samples (WLASL, Kaggle sign) → chỉ CE loss.

    Lý do: CTC cần input_length >> target_length.
    WLASL video chỉ có 1 ký hiệu (T rất ngắn) → CTC sẽ throw error
    hoặc gây gradient explosion.
    """
    gloss_ids = batch["gloss_ids"]      # (B, Lg)
    gloss_lengths = batch["gloss_lengths"]  # (B,)
    dataset_modes = batch.get("dataset_mode", ["continuous"] * gloss_ids.size(0))

    # Mask tách continuous vs isolated
    is_continuous = torch.tensor(
        [m == "continuous" for m in dataset_modes],
        device=gloss_ids.device, dtype=torch.bool
    )

    loss_ctc = torch.tensor(0.0, device=gloss_ids.device)
    loss_ce = torch.tensor(0.0, device=gloss_ids.device)

    # ── CTC Loss (chỉ cho continuous samples) ──
    n_continuous = is_continuous.sum().item()
    if n_continuous > 0:
        # Lọc continuous samples
        c_idx = is_continuous.nonzero(as_tuple=True)[0]
        log_probs = outputs["gloss_log_probs"][:, c_idx, :]   # (T, n_cont, G)
        input_lengths = outputs["encoder_lengths"][c_idx]      # (n_cont,)
        targets = gloss_ids[c_idx]                             # (n_cont, Lg)
        target_lengths = gloss_lengths[c_idx]                  # (n_cont,)

        # Đảm bảo input_length >= target_length (CTC constraint)
        valid = input_lengths >= target_lengths
        if valid.sum() > 0:
            loss_ctc = criterion_ctc(
                log_probs[:, valid, :],
                targets[valid],
                input_lengths[valid],
                target_lengths[valid],
            )

    # ── CE Loss (tất cả samples) ──
    text_logits = outputs.get("text_logits")
    if text_logits is not None:
        B, Lw, V = text_logits.shape
        text_ids = batch["text_ids"]
        loss_ce = criterion_ce(
            text_logits.reshape(B * Lw, V),
            text_ids.reshape(B * Lw),
        )

    total_loss = config.lambda_ctc * loss_ctc + config.lambda_ce * loss_ce
    return total_loss, loss_ctc, loss_ce


# ══════════════════════════════════════════════════════════════
#  TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, scaler, scheduler,
                    criterion_ctc, criterion_ce, config, epoch, logger,
                    drive_saver: TimedDriveSaver = None,
                    output_dir: str = "outputs/"):
    model.train()
    losses = AverageMeter()
    ctc_losses = AverageMeter()
    ce_losses = AverageMeter()

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        keypoints = batch["keypoints"].to(config.device)
        gloss_ids = batch["gloss_ids"].to(config.device)
        text_ids = batch["text_ids"].to(config.device)
        gloss_lengths = batch["gloss_lengths"].to(config.device)
        keypoint_lengths = batch["keypoint_lengths"].to(config.device)
        # Giữ dataset_mode trên CPU (list of strings)

        with autocast('cuda', enabled=config.use_amp):
            outputs = model(
                keypoints=keypoints,
                keypoint_lengths=keypoint_lengths,
                gloss_targets=gloss_ids,
                text_targets=text_ids,
            )

            # Dataset-mode aware loss routing
            batch["gloss_ids"] = gloss_ids
            batch["gloss_lengths"] = gloss_lengths
            batch["text_ids"] = text_ids
            loss, loss_ctc, loss_ce = compute_loss(
                outputs, batch, criterion_ctc, criterion_ce, config
            )
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # ── Timed Drive backup (per optimizer step) ──
            if drive_saver and drive_saver.should_save():
                latest_path = Path(output_dir) / "checkpoint_latest.pt"
                if latest_path.exists():
                    drive_saver.save(str(latest_path), "timed")

        B_size = keypoints.size(0)
        losses.update(loss.item() * config.gradient_accumulation_steps, B_size)
        ctc_losses.update(loss_ctc.item(), B_size)
        ce_losses.update(loss_ce.item(), B_size)

        if step % config.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(dataloader)}] "
                f"Loss: {losses.avg:.4f} CTC: {ctc_losses.avg:.4f} "
                f"CE: {ce_losses.avg:.4f} LR: {lr:.2e}"
            )

    return {"loss": losses.avg, "ctc_loss": ctc_losses.avg, "ce_loss": ce_losses.avg}


# ══════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, dataloader, gloss_vocab, text_vocab, config, logger, split="val"):
    model.eval()
    all_gloss_preds, all_gloss_gts = [], []
    all_text_preds, all_text_gts = [], []

    criterion_ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=text_vocab.pad_id)

    for batch in dataloader:
        keypoints = batch["keypoints"].to(config.device)
        gloss_ids = batch["gloss_ids"].to(config.device)
        text_ids = batch["text_ids"].to(config.device)
        gloss_lengths = batch["gloss_lengths"].to(config.device)
        keypoint_lengths = batch["keypoint_lengths"].to(config.device)

        with autocast('cuda', enabled=config.use_amp):
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


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    config = get_config(args.config)
    config.output_dir = args.output_dir
    config.drive_backup = args.drive_backup

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config.output_dir)
    set_seed(config.seed)

    logger.info("=" * 60)
    logger.info("  UPGRADED HST-GNN v2 TRAINING")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Phase: {args.phase}  |  Device: {config.device}")
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
        temporal_window_size=getattr(config, "temporal_window_size", 5),
        decoder_name=config.decoder_name,
        dropout=config.dropout,
        use_gradient_checkpointing=config.gradient_checkpointing,
    ).to(config.device)

    # Áp dụng training phase
    set_training_phase(model, args.phase, logger)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params/1e6:.2f}M | Trainable: {train_params/1e6:.2f}M")

    # ── Loss ──────────────────────────────────────────────────
    criterion_ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=text_vocab.pad_id)

    # ── Optimizer ─────────────────────────────────────────────
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

    # ── Timed Drive saver ─────────────────────────────────────
    drive_saver = TimedDriveSaver(
        drive_path=args.drive_backup,
        interval_min=args.save_interval_min,
    )

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 1
    best_wer = float("inf")
    if args.resume:
        start_epoch, best_wer = load_checkpoint(
            args.resume, model, optimizer, scheduler, logger
        )
    else:
        ckpt_path = Path(config.output_dir) / "checkpoint_latest.pt"
        if ckpt_path.exists():
            start_epoch, best_wer = load_checkpoint(
                str(ckpt_path), model, optimizer, scheduler, logger
            )

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
            criterion_ctc, criterion_ce, config, epoch, logger,
            drive_saver=drive_saver,
            output_dir=config.output_dir,
        )

        val_metrics = evaluate(
            model, val_loader, gloss_vocab, text_vocab, config, logger, "val"
        )

        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch} done in {elapsed:.1f}s")

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
            drive_backup=None,  # Handled by TimedDriveSaver below
        )

        # Cuối mỗi epoch: force save lên Drive
        latest_path = Path(config.output_dir) / "checkpoint_latest.pt"
        drive_saver.force_save(str(latest_path), f"epoch_{epoch:03d}")
        if is_best:
            best_path = Path(config.output_dir) / "checkpoint_best.pt"
            drive_saver.save(str(best_path), "best")

        # Early stopping
        if early_stopping(val_metrics["wer"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Final Test ────────────────────────────────────────────
    logger.info("Loading best checkpoint for final test...")
    best_ckpt = Path(config.output_dir) / "checkpoint_best.pt"
    if best_ckpt.exists():
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
