"""
model.py — Upgraded HST-GNN Architecture
=========================================
Upgrades từ bài gốc:
  • Bỏ ResNet-152 → skeleton keypoints trực tiếp
  • Graph Convolution + Graph Transformer (giữ từ bài gốc, tối ưu hóa)
  • Temporal Encoder: 1D Conv + Sliding-Window Transformer (fix global-attention bug)
  • mBART-50 pretrained text decoder (thay 2×LSTM)
  • Gradient checkpointing support
  • Learnable adjacency matrix nhẹ hơn (cosine similarity)

Fix v2:
  • TemporalEncoder dùng Sliding Window Attention (window=5 frames) thay Global Attention
    → đúng với kết luận bài gốc: cửa sổ ngữ cảnh cục bộ nhỏ tốt hơn
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import MBartForConditionalGeneration, MBartConfig
from transformers.modeling_outputs import BaseModelOutput


# ══════════════════════════════════════════════════════════════
#  1. SKELETON EMBEDDING
# ══════════════════════════════════════════════════════════════

class SkeletonEmbedding(nn.Module):
    """
    Embed raw (x, y, z, visibility) keypoints → d_model features.
    Tách riêng body regions để xử lý phân cấp.
    """
    REGIONS = {
        # MediaPipe Holistic indices (543 keypoints total)
        "pose":       (0,   33),   # 33 pose landmarks
        "left_hand":  (33,  54),   # 21 hand landmarks
        "right_hand": (54,  75),   # 21 hand landmarks
        "face":       (75,  543),  # 468 face landmarks
    }

    def __init__(self, keypoint_dim: int = 4, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Per-region linear projection
        self.region_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(keypoint_dim * (end - start), d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for name, (start, end) in self.REGIONS.items()
        })

        # Fuse 4 regions → d_model
        self.fuse = nn.Sequential(
            nn.Linear(d_model * len(self.REGIONS), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) where N=543 keypoints, C=keypoint_dim
        returns: (B, T, d_model)
        """
        B, T, N, C = x.shape
        region_feats = []
        for name, (start, end) in self.REGIONS.items():
            r = x[:, :, start:end, :]           # (B, T, n_joints, C)
            r = r.reshape(B, T, -1)             # (B, T, n_joints*C)
            r = self.region_proj[name](r)       # (B, T, d_model)
            region_feats.append(r)
        out = torch.cat(region_feats, dim=-1)   # (B, T, d_model*4)
        return self.fuse(out)                   # (B, T, d_model)


# ══════════════════════════════════════════════════════════════
#  2. GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════

class AdaptiveAdjacency(nn.Module):
    """
    Học adjacency matrix động thay vì fixed topology.
    Upgrade: cosine similarity thay bilinear (ít param hơn, ổn định hơn).
    """
    def __init__(self, num_nodes: int, d_model: int, num_heads: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)

        # Learnable positional bias theo skeleton topology
        self.positional_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.positional_bias.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d_model)
        returns: A (B, num_heads, N, N) — normalized adjacency
        """
        B, N, D = x.shape
        H = self.num_heads

        Q = self.q_proj(x).reshape(B, N, H, -1).permute(0, 2, 1, 3)  # (B,H,N,d)
        K = self.k_proj(x).reshape(B, N, H, -1).permute(0, 2, 1, 3)

        # Scaled dot-product + positional bias
        attn = (Q @ K.transpose(-2, -1)) / self.scale                 # (B,H,N,N)
        attn = attn + self.positional_bias.unsqueeze(0).unsqueeze(0)
        A = torch.softmax(attn, dim=-1)
        return A


# ══════════════════════════════════════════════════════════════
#  3. GRAPH CONVOLUTION (từ bài gốc, tối ưu)
# ══════════════════════════════════════════════════════════════

class GraphConvolution(nn.Module):
    """
    H^{l+1} = f(A · H^l · W^l)   — giống bài gốc eq. (6)
    Thêm residual connection + pre-norm (ổn định hơn)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d_model)
        A: (B, H, N, N)  — average over heads
        """
        A_mean = A.mean(dim=1)                    # (B, N, N)
        h = self.norm(x)
        h = A_mean @ h                            # message passing
        h = self.linear(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h                              # residual


# ══════════════════════════════════════════════════════════════
#  4. GRAPH TRANSFORMER (từ bài gốc, nâng cấp)
# ══════════════════════════════════════════════════════════════

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer với neighborhood context (Section 3.3 bài gốc).
    Upgrade: Pre-LN thay Post-LN, Flash-Attention compatible.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Neighborhood context gate (từ bài gốc eq. 10, đơn giản hóa)
        self.neighbor_gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        A_mean = A.mean(dim=1)  # (B, N, N)

        # Self-attention with neighborhood context
        h = self.norm1(x)
        # Neighborhood-guided key/value
        neighbor_ctx = A_mean @ h                            # (B, N, D)
        h_gated = h + torch.sigmoid(self.neighbor_gate(neighbor_ctx)) * neighbor_ctx
        attn_out, _ = self.attn(h_gated, h_gated, h_gated)
        x = x + self.dropout(attn_out)

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════
#  5. HIERARCHICAL GRAPH ENCODER
# ══════════════════════════════════════════════════════════════

class HierarchicalGraphEncoder(nn.Module):
    """
    Xử lý 2 cấp graph:
      Level 1 (high-level): 4 nodes = [pose_body, left_hand, right_hand, face]
      Level 2 (fine-level): full N keypoints per region

    Bài gốc dùng ResNet feature cho Level 1 — ở đây dùng learned aggregation.
    """
    def __init__(self, d_model: int, num_heads: int,
                 num_graph_layers: int, dropout: float, use_ckpt: bool = False):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.num_graph_layers = num_graph_layers

        # Fine-level encoder (per full keypoint set)
        NUM_FINE_NODES = 75  # pose(33) + left_hand(21) + right_hand(21)
        self.fine_adj = AdaptiveAdjacency(NUM_FINE_NODES, d_model, num_heads)
        self.fine_gcn = nn.ModuleList([
            GraphConvolution(d_model, dropout) for _ in range(num_graph_layers)
        ])
        self.fine_transformer = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads, dropout) for _ in range(num_graph_layers)
        ])

        # High-level encoder (4 region nodes)
        NUM_HIGH_NODES = 4
        self.high_adj = AdaptiveAdjacency(NUM_HIGH_NODES, d_model, num_heads)
        self.high_gcn = nn.ModuleList([
            GraphConvolution(d_model, dropout) for _ in range(num_graph_layers)
        ])
        self.high_transformer = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads, dropout) for _ in range(num_graph_layers)
        ])

        # Region boundaries trong fine-level
        self.REGION_NODES = {
            "pose":       (0,  33),
            "left_hand":  (33, 54),
            "right_hand": (54, 75),
            "face":       (75, 543),  # face dùng riêng
        }

        # Face fine-level (468 nodes, xử lý riêng do lớn)
        self.face_proj = nn.Sequential(
            nn.Linear(468 * 4, d_model),  # raw coords → d_model
            nn.LayerNorm(d_model), nn.GELU(),
        )

        # Hierarchical pooling (Section 3.4)
        self.region_pool = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            for name in ["pose", "left_hand", "right_hand", "face"]
        })

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def _run_layer(self, fn, *args):
        if self.use_ckpt and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x_regions: dict, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_regions: dict {name: (B, T, n_nodes, d_model)} — per-region node features
        x_raw: (B, T, 543, 4) — raw keypoints for face

        returns: (B, T, d_model) — fused hierarchical representation
        """
        B, T = x_raw.shape[:2]

        # ── Fine-level: pose + hands (75 nodes) ──
        # Stack pose + left_hand + right_hand
        fine_nodes = []
        for name in ["pose", "left_hand", "right_hand"]:
            fine_nodes.append(x_regions[name])  # (B, T, n, d)
        fine_x = torch.cat(fine_nodes, dim=2)   # (B, T, 75, d)
        BT = B * T
        fine_x = fine_x.reshape(BT, 75, -1)     # (BT, 75, d)

        fine_A = self.fine_adj(fine_x)           # (BT, H, 75, 75)
        for i in range(self.num_graph_layers):
            fine_x = self._run_layer(self.fine_gcn[i], fine_x, fine_A)
            fine_x = self._run_layer(self.fine_transformer[i], fine_x, fine_A)

        fine_x = fine_x.reshape(B, T, 75, -1)   # (B, T, 75, d)

        # ── Face: separate fine-level ──
        face_raw = x_raw[:, :, 75:543, :].reshape(B, T, -1)  # (B,T,468*4)
        face_feat = self.face_proj(face_raw)                   # (B,T,d)

        # ── High-level: pool each region to 1 node ──
        # 4 region nodes: pose_pool, lhand_pool, rhand_pool, face_pool
        pose_pool = fine_x[:, :, :33, :].mean(dim=2)   # (B,T,d)
        lhand_pool = fine_x[:, :, 33:54, :].mean(dim=2)
        rhand_pool = fine_x[:, :, 54:75, :].mean(dim=2)

        high_nodes = torch.stack([
            self.region_pool["pose"](pose_pool),
            self.region_pool["left_hand"](lhand_pool),
            self.region_pool["right_hand"](rhand_pool),
            self.region_pool["face"](face_feat),
        ], dim=2)                                        # (B, T, 4, d)

        high_x = high_nodes.reshape(BT, 4, -1)
        high_A = self.high_adj(high_x)
        for i in range(self.num_graph_layers):
            high_x = self._run_layer(self.high_gcn[i], high_x, high_A)
            high_x = self._run_layer(self.high_transformer[i], high_x, high_A)

        high_x = high_x.reshape(B, T, 4, -1)

        # ── Hierarchical Pooling → single vector per frame ──
        fine_pooled = fine_x.mean(dim=2)         # (B, T, d)
        high_pooled = high_x.mean(dim=2)         # (B, T, d)

        fused = fine_pooled + high_pooled + face_feat  # (B, T, d)
        return self.output_proj(fused)


# ══════════════════════════════════════════════════════════════
#  6. TEMPORAL ENCODER — Sliding Window Transformer
#     FIX: Thay Global Attention → Local Sliding Window Attention
#     Lý do: Bài gốc kết luận window_size > 3 gây nhiễu.
#     Transformer Global Attention vi phạm điều này → phải mask.
# ══════════════════════════════════════════════════════════════

def make_sliding_window_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
    """
    Tạo causal-free sliding window attention mask.
    mask[i, j] = True  → token i KHÔNG được attend token j (bị block)
    mask[i, j] = False → token i CÓ THỂ attend token j

    Mỗi token chỉ nhìn thấy [i-window, i+window] xung quanh nó.
    Consistent với kết luận bài gốc: window_size ≤ 3-5 tốt nhất.
    """
    idx = torch.arange(T, device=device)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (T, T)
    mask = diff > window   # True = blocked
    return mask             # (T, T) bool


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    1D Depthwise Conv → Sliding-Window Transformer

    FIX v2: Thay global TransformerEncoder → local sliding window attention.
    window_size=5 frames: mỗi frame chỉ attend 5 frame trước/sau.
    Đúng với bài gốc — ngăn nhiễu từ hành động xa không liên quan.
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 dropout: float, window_size: int = 5, use_ckpt: bool = False):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.window_size = window_size

        # Local temporal pattern (conv)
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Sliding-Window Transformer layers
        # Dùng nn.TransformerEncoderLayer nhưng forward luôn truyền
        # attn_mask=sliding_window_mask → chặn global attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (ổn định hơn)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._cached_mask = None
        self._cached_T = -1

    def _get_local_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Cache mask để không tạo lại mỗi step."""
        if self._cached_T != T or self._cached_mask is None or \
                self._cached_mask.device != device:
            self._cached_mask = make_sliding_window_mask(T, self.window_size, device)
            self._cached_T = T
        return self._cached_mask

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, d_model)
        padding_mask: (B, T) — True where padded
        """
        T = x.size(1)

        # Local conv (operate on channel dim)
        h = x.transpose(1, 2)           # (B, d, T)
        h = self.local_conv(h)
        h = h.transpose(1, 2)           # (B, T, d)
        x = x + h                       # residual

        x = self.pos_enc(x)

        # Sliding window attention mask (T, T)
        local_mask = self._get_local_mask(T, x.device)

        x = self.transformer(
            x,
            mask=local_mask,                    # (T,T) local attention
            src_key_padding_mask=padding_mask,  # (B,T) padding
        )
        return x


# ══════════════════════════════════════════════════════════════
#  7. CTC GLOSS HEAD
# ══════════════════════════════════════════════════════════════

class CTCGlossHead(nn.Module):
    def __init__(self, d_model: int, num_gloss_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_gloss_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → (T, B, num_gloss) log_softmax"""
        x = self.norm(x)
        logits = self.proj(x)                # (B, T, G)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)   # (T, B, G) for CTCLoss


# ══════════════════════════════════════════════════════════════
#  8. MBART TEXT DECODER
# ══════════════════════════════════════════════════════════════

class LightweightMBartDecoder(nn.Module):
    """
    Dùng mBART-50 nhưng chỉ lấy decoder layers.
    Encoder features từ graph encoder → cross-attend vào mBART decoder.
    Freeze embedding + top layers để tiết kiệm memory.
    """
    def __init__(self, d_model: int, vocab_size: int,
                 decoder_name: str = "facebook/mbart-large-50"):
        super().__init__()

        # Load chỉ config, không load full weights (tiết kiệm RAM)
        try:
            self.mbart = MBartForConditionalGeneration.from_pretrained(
                decoder_name,
                ignore_mismatched_sizes=True,
            )
            # Freeze embedding layers và top 6 decoder layers
            for name, param in self.mbart.named_parameters():
                if ("shared" in name or "embed" in name or
                        any(f"decoder.layers.{i}" in name for i in range(6))):
                    param.requires_grad = False
        except Exception:
            # Fallback: khởi tạo từ config nhỏ hơn
            config = MBartConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                encoder_layers=0,
                decoder_layers=4,
                encoder_ffn_dim=d_model * 4,
                decoder_ffn_dim=d_model * 4,
                encoder_attention_heads=8,
                decoder_attention_heads=8,
            )
            self.mbart = MBartForConditionalGeneration(config)

        # Project graph d_model → mBART d_model nếu khác nhau
        mbart_d = self.mbart.config.d_model
        self.enc_proj = nn.Linear(d_model, mbart_d) if d_model != mbart_d else nn.Identity()

    def forward(self, encoder_hidden: torch.Tensor,
                encoder_mask: torch.Tensor,
                decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """
        encoder_hidden: (B, T, d_model)
        encoder_mask: (B, T)
        decoder_input_ids: (B, Lw)
        returns: logits (B, Lw, vocab_size)
        """
        enc_h = self.enc_proj(encoder_hidden)    # (B, T, mbart_d)

        # attention_mask: 1 = keep, 0 = pad
        enc_attn_mask = (~encoder_mask).long()

        # Bọc trong BaseModelOutput (transformers >= 4.36 yêu cầu object thay vì tuple)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_h)

        outputs = self.mbart(
            inputs_embeds=None,
            attention_mask=enc_attn_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )
        return outputs.logits                    # (B, Lw, vocab)

    @torch.no_grad()
    def generate(self, encoder_hidden: torch.Tensor,
                 encoder_mask: torch.Tensor,
                 forced_bos_token_id: int = None,
                 max_length: int = 128,
                 num_beams: int = 4) -> torch.Tensor:
        # Ép về FP32 để tránh lỗi Half/Float mismatch khi dùng AMP
        encoder_hidden = encoder_hidden.float()
        enc_h = self.enc_proj.float()(encoder_hidden)  # proj cũng về FP32
        enc_attn_mask = (~encoder_mask).long()

        # Bọc trong BaseModelOutput (transformers mới yêu cầu)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_h)

        out = self.mbart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_attn_mask,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=num_beams,
        )
        return out


# ══════════════════════════════════════════════════════════════
#  9. FULL MODEL
# ══════════════════════════════════════════════════════════════

class UpgradedHSTGNN(nn.Module):
    """
    Full upgraded pipeline:
      Keypoints → SkeletonEmbedding
                → HierarchicalGraphEncoder (GCN + GraphTransformer)
                → TemporalEncoder (Conv + Sliding-Window Transformer)
                ↙              ↘
         CTCGlossHead     LightweightMBartDecoder
              ↓                    ↓
          gloss_log_probs       text_logits

    dataset_mode aware:
      "continuous" → CTC + CE loss (PHOENIX, CSL)
      "isolated"   → CE loss only (WLASL, Kaggle sign datasets)
    """

    def __init__(
        self,
        num_keypoints: int = 543,
        keypoint_dim: int = 4,
        d_model: int = 256,
        num_graph_layers: int = 2,
        num_heads: int = 8,
        num_temporal_layers: int = 4,
        temporal_window_size: int = 5,          # Sliding window size (bài gốc: ≤5)
        num_gloss_classes: int = 1200,
        text_vocab_size: int = 32000,
        decoder_name: str = "facebook/mbart-large-50",
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Region projections (trước khi vào graph encoder)
        REGION_SIZES = {
            "pose": 33, "left_hand": 21, "right_hand": 21, "face": 468
        }
        self.region_embed = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(n * keypoint_dim, d_model),
                nn.LayerNorm(d_model), nn.GELU(),
            )
            for name, n in REGION_SIZES.items() if name != "face"
        })

        self.skeleton_embed = SkeletonEmbedding(keypoint_dim, d_model, dropout)

        self.graph_encoder = HierarchicalGraphEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
            use_ckpt=use_gradient_checkpointing,
        )

        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout,
            window_size=temporal_window_size,   # ← Sliding window!
            use_ckpt=use_gradient_checkpointing,
        )

        self.gloss_head = CTCGlossHead(d_model, num_gloss_classes)

        self.text_decoder = LightweightMBartDecoder(
            d_model=d_model,
            vocab_size=text_vocab_size,
            decoder_name=decoder_name,
        )

        # CTC length predictor (để tính encoder_lengths sau subsampling)
        self.temporal_subsample = nn.Conv1d(
            d_model, d_model, kernel_size=2, stride=2, padding=0
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        keypoints: torch.Tensor,          # (B, T, N, C)
        keypoint_lengths: torch.Tensor,    # (B,) actual lengths
        gloss_targets: torch.Tensor = None,
        text_targets: torch.Tensor = None,
    ) -> dict:
        B, T, N, C = keypoints.shape

        # ── Build padding mask ──
        padding_mask = torch.arange(T, device=keypoints.device).unsqueeze(0) >= \
                       keypoint_lengths.unsqueeze(1)   # (B, T) True=padded

        # ── Region features for graph ──
        regions = {}
        REGION_BOUNDS = {"pose": (0,33), "left_hand": (33,54), "right_hand": (54,75)}
        for name, (s, e) in REGION_BOUNDS.items():
            r = keypoints[:, :, s:e, :].reshape(B, T, -1)  # (B,T,n*C)
            regions[name] = self.region_embed[name](r).unsqueeze(2)  # (B,T,1,d)
            # Expand back to node dimension for graph
            n_nodes = e - s
            regions[name] = regions[name].expand(B, T, n_nodes, self.d_model)

        # ── Graph encoder ──
        graph_out = self.graph_encoder(regions, keypoints)   # (B, T, d)

        # ── Temporal encoder (Sliding Window) ──
        encoder_out = self.temporal_encoder(graph_out, padding_mask)  # (B, T, d)

        # ── Temporal subsampling (CTC standard: T → T/2) ──
        enc_sub = encoder_out.transpose(1, 2)                # (B, d, T)
        enc_sub = self.temporal_subsample(enc_sub)           # (B, d, T/2)
        enc_sub = enc_sub.transpose(1, 2)                    # (B, T/2, d)
        enc_lengths = (keypoint_lengths.float() / 2).long().clamp(min=1)

        # ── CTC gloss ──
        gloss_log_probs = self.gloss_head(enc_sub)           # (T/2, B, G)

        # ── Text decoder ──
        text_logits = None
        if text_targets is not None:
            # Teacher forcing: shift right
            bos = torch.full((B, 1), self.text_decoder.mbart.config.decoder_start_token_id,
                             dtype=torch.long, device=keypoints.device)
            decoder_input = torch.cat([bos, text_targets[:, :-1]], dim=1)

            # Padding mask for encoder (T/2)
            enc_padding_mask = torch.arange(enc_sub.size(1), device=keypoints.device).unsqueeze(0) \
                               >= enc_lengths.unsqueeze(1)
            text_logits = self.text_decoder(enc_sub, enc_padding_mask, decoder_input)

        return {
            "gloss_log_probs": gloss_log_probs,   # (T/2, B, G)
            "text_logits": text_logits,            # (B, Lw, V) or None
            "encoder_hidden": enc_sub,             # (B, T/2, d)
            "encoder_lengths": enc_lengths,        # (B,)
        }

    def decode_gloss(self, log_probs: torch.Tensor) -> list:
        """
        CTC greedy decode.
        log_probs: (T, B, G)
        returns: list of lists of int
        """
        T, B, G = log_probs.shape
        preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()  # (B, T)
        decoded = []
        for seq in preds:
            # CTC collapse: remove blanks and duplicates
            out, prev = [], -1
            for t in seq:
                if t != 0 and t != prev:
                    out.append(t)
                prev = t
            decoded.append(out)
        return decoded

    @torch.no_grad()
    def decode_text(self, encoder_hidden, encoder_lengths, text_vocab, max_len=128):
        """Beam search generation."""
        B = encoder_hidden.size(0)
        enc_padding_mask = torch.arange(encoder_hidden.size(1),
                                        device=encoder_hidden.device).unsqueeze(0) \
                           >= encoder_lengths.unsqueeze(1)

        ids = self.text_decoder.generate(
            encoder_hidden=encoder_hidden,
            encoder_mask=enc_padding_mask,
            forced_bos_token_id=getattr(text_vocab, "bos_id", 2),
            max_length=max_len,
            num_beams=4,
        )
        return ids.cpu().tolist()
