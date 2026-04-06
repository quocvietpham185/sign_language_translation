"""
scheduler.py — Warmup + Cosine Annealing LR Scheduler
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup → Cosine decay.
    Ổn định hơn StepLR, tốt hơn ReduceOnPlateau cho Transformer.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.01, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]
