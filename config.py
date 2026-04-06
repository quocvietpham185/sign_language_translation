"""
config.py — Configuration Management
"""

import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────
    train_data_path: str = "data/train.json"
    val_data_path: str = "data/val.json"
    test_data_path: str = "data/test.json"
    gloss_vocab_path: str = "data/gloss_vocab.json"
    text_vocab_path: str = "data/text_vocab.json"
    num_keypoints: int = 543
    keypoint_dim: int = 4
    max_seq_len: int = 512
    max_text_len: int = 128
    num_workers: int = 2

    # ── Model ─────────────────────────────────────────────────
    d_model: int = 256
    num_graph_layers: int = 2
    num_temporal_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    decoder_name: str = "facebook/mbart-large-50"
    gradient_checkpointing: bool = True

    # ── Training ──────────────────────────────────────────────
    epochs: int = 80
    batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4   # effective batch = 32
    lr: float = 1e-3
    pretrained_lr_scale: float = 0.1       # LR cho pretrained mBART
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    patience: int = 15
    use_amp: bool = True
    seed: int = 42

    # ── Loss weights ──────────────────────────────────────────
    lambda_ctc: float = 0.5
    lambda_ce: float = 0.5

    # ── Augmentation ──────────────────────────────────────────
    aug_flip_prob: float = 0.5
    aug_noise_std: float = 0.01
    aug_scale_range: Tuple[float, float] = (0.85, 1.15)
    aug_rotation: float = 15.0
    aug_time_stretch: Tuple[float, float] = (0.8, 1.2)
    aug_frame_drop: float = 0.1
    aug_joint_drop: float = 0.05

    # ── Logging ───────────────────────────────────────────────
    log_interval: int = 50
    output_dir: str = "outputs/"
    drive_backup: Optional[str] = None

    # ── Device (auto) ─────────────────────────────────────────
    device: str = field(init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_config(yaml_path: str = None) -> Config:
    config = Config()
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            for k, v in overrides.items():
                if hasattr(config, k):
                    setattr(config, k, v)
    return config
