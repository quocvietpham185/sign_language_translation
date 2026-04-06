# 🤟 Upgraded HST-GNN — Sign Language Translation

Nâng cấp toàn diện từ: **[arxiv 2111.07258](https://arxiv.org/abs/2111.07258)** — Kan et al., 2021  
*"Sign Language Translation with Hierarchical Spatio-Temporal Graph Neural Network"*

---

## So sánh với bài gốc

| Component | Bài gốc (HST-GNN) | Bản nâng cấp |
|---|---|---|
| Feature extraction | ResNet-152 + TVL1-flow (nặng) | **MediaPipe Holistic keypoints** (free, offline) |
| Adjacency matrix | Bilinear transform (nhiều param) | **Cosine attention + learnable bias** (ổn định hơn) |
| Temporal modeling | 2×LSTM (sequential) | **1D Conv + Transformer** (song song, nhanh hơn) |
| Text decoder | 2-stage LSTM | **mBART-50 pretrained** (BLEU tốt hơn rõ rệt) |
| Graph normalization | Post-LN | **Pre-LN** (ổn định training) |
| LR schedule | Fixed 0.001 | **Warmup + Cosine Decay** |
| Data augmentation | Không có | **7 loại** (flip, noise, scale, rotate, time stretch, frame drop, joint drop) |
| Mixed Precision | Không | **AMP** (2× faster, giảm VRAM) |
| Gradient checkpointing | Không | **Có** (train được trên Colab free) |
| Dataset | PHOENIX + CSL | **Multi-dataset** (PHOENIX, WLASL, CSL, Kaggle) |
| Tham số huấn luyện | ~80-100M | **~15-20M** (mBART frozen phần lớn) |

---

## Cấu trúc project

```
sign_language/
├── train.py           # Main training script
├── model.py           # Upgraded HST-GNN architecture
├── dataset.py         # Dataset + 7 augmentations + kagglehub loader
├── config.py          # Configuration dataclass
├── vocabulary.py      # Vocabulary builder + encoder/decoder
├── utils.py           # Metrics (WER, BLEU), checkpointing, logging
├── scheduler.py       # Warmup + Cosine LR scheduler
├── configs/
│   ├── phoenix.yaml   # PHOENIX-2014-T (full, server/Colab Pro)
│   └── colab.yaml     # Colab free (auto-generated trong notebook)
├── train_colab.ipynb  # Google Colab notebook (all-in-one)
└── requirements.txt
```

---

## Kiến trúc Model

```
Input: MediaPipe keypoints (T × 543 × 4)
         │  [x, y, z, visibility] per keypoint
         │
         ▼
┌─────────────────────────────────────────────┐
│         SKELETON EMBEDDING                  │
│  Pose(33) + L.Hand(21) + R.Hand(21) → d    │
│  Face(468) → d   (separate path)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│    HIERARCHICAL GRAPH ENCODER               │
│                                             │
│  Fine-level (75 nodes):                     │
│    Pose + Hand joints → GCN → GraphTransf   │
│                                             │
│  High-level (4 nodes):                      │
│    [Pose, L.Hand, R.Hand, Face] pooled      │
│    → GCN → GraphTransformer                 │
│                                             │
│  Hierarchical Pooling → (B, T, d)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│       TEMPORAL ENCODER                      │
│  1D DepthwiseConv (local) + Positional Enc  │
│  + Transformer Layers (global)              │
│  → (B, T/2, d)   [2× subsampling]          │
└──────────┬──────────────────┬───────────────┘
           │                  │
           ▼                  ▼
┌──────────────┐    ┌─────────────────────┐
│  CTC GLOSS   │    │  mBART-50 DECODER   │
│     HEAD     │    │  (pretrained,       │
│              │    │   partially frozen) │
│ → gloss_log  │    │  → text_logits      │
│    _probs    │    │                     │
└──────────────┘    └─────────────────────┘
       │                      │
       ▼                      ▼
   L_CTC loss             L_CE loss
       └──────────┬───────────┘
                  ▼
          L = 0.5·CTC + 0.5·CE
```

---

## Cách dùng

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

#### Option A: Dùng dataset từ Kaggle
```python
from dataset import load_kaggle_sign_dataset
load_kaggle_sign_dataset("datamunge/sign-language-mnist", "data/train.json")
```

#### Option B: Extract keypoints từ video PHOENIX
```python
from dataset import batch_extract_keypoints
batch_extract_keypoints(
    video_dir="phoenix/videos/",
    output_dir="phoenix/keypoints/",
    annotation_file="phoenix/annotations.json",
    num_workers=8
)
```

### 3. Build vocabularies
```python
from vocabulary import Vocabulary
gloss_vocab = Vocabulary.build_from_json("data/train.json", "gloss")
gloss_vocab.save("data/gloss_vocab.json")
```

### 4. Train
```bash
# PHOENIX (full)
python train.py --config configs/phoenix.yaml

# Resume từ checkpoint
python train.py --config configs/phoenix.yaml --resume outputs/checkpoint_latest.pt

# Eval only
python train.py --config configs/phoenix.yaml --resume outputs/checkpoint_best.pt --eval_only
```

### 5. Google Colab
Mở `train_colab.ipynb` → Run all cells

---

## Chiến lược training trên Colab Free

```
Session 1 (~2h): epochs 1-15, lưu checkpoint lên Drive
Session 2 (~2h): resume từ Drive, epochs 16-30
Session 3 (~2h): resume từ Drive, epochs 31-50
...
```

Khi hết GPU free:
- **Kaggle Notebooks**: 30h GPU/tuần (P100), miễn phí
- **Lightning.AI Studio**: 22h GPU/tháng, persistent workspace  
- **Colab Pro**: $10/tháng, A100 GPU

---

## Kết quả mục tiêu

| Metric | Bài gốc | Mục tiêu |
|---|---|---|
| WER (Phoenix test) | 19.5% | **< 18.5%** |
| BLEU-4 (Phoenix test) | 22.3 | **> 24.0** |
| Train time (T4) | ~48h | **~20h** |
| VRAM required | ~24GB | **~10GB** |

---

## Citation

```bibtex
@article{kan2021sign,
  title={Sign Language Translation with Hierarchical Spatio-Temporal Graph Neural Network},
  author={Kan, Jichao and Hu, Kun and Hagenbuchner, Markus and others},
  journal={arXiv preprint arXiv:2111.07258},
  year={2021}
}
```
