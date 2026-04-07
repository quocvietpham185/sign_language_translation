"""
dataset.py — Sign Language Dataset
====================================
Hỗ trợ:
  • PHOENIX-2014-T (đã pre-extracted MediaPipe keypoints)
  • WLASL (từ Kaggle qua kagglehub)
  • CSL
  • Bất kỳ dataset nào theo cùng format JSON/CSV

Data Augmentation (tất cả trên keypoints, không cần GPU):
  1. Random horizontal flip (trái ↔ phải)
  2. Gaussian noise injection
  3. Random scaling (mô phỏng khoảng cách camera)
  4. Random rotation (±15°)
  5. Temporal stretch / speed perturbation
  6. Random frame dropout (occlusion simulation)
  7. Random joint dropout (keypoint visibility loss)
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════

class KeypointAugmentor:
    """
    Tất cả augmentation áp dụng trên numpy array (T, N, C).
    C = (x, y, z, visibility)
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_std: float = 0.01,
        scale_range: tuple = (0.85, 1.15),
        rotation_range: float = 15.0,   # degrees
        time_stretch_range: tuple = (0.8, 1.2),
        frame_drop_prob: float = 0.1,
        joint_drop_prob: float = 0.05,
    ):
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.time_stretch_range = time_stretch_range
        self.frame_drop_prob = frame_drop_prob
        self.joint_drop_prob = joint_drop_prob

        # MediaPipe Holistic flip pairs (left ↔ right symmetry)
        # Pose landmarks: 11↔12(shoulders), 13↔14(elbows), 15↔16(wrists), etc.
        self.POSE_FLIP_PAIRS = [
            (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
            (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
        ]

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """keypoints: (T, N, C)"""
        # Apply augmentations randomly
        if random.random() < self.flip_prob:
            keypoints = self._horizontal_flip(keypoints)
        if random.random() < 0.7:
            keypoints = self._add_noise(keypoints)
        if random.random() < 0.5:
            keypoints = self._random_scale(keypoints)
        if random.random() < 0.3:
            keypoints = self._random_rotation(keypoints)
        if random.random() < 0.4:
            keypoints = self._time_stretch(keypoints)
        if random.random() < 0.3:
            keypoints = self._frame_dropout(keypoints)
        if random.random() < 0.2:
            keypoints = self._joint_dropout(keypoints)
        return keypoints

    def _horizontal_flip(self, kps: np.ndarray) -> np.ndarray:
        """Flip x coordinate và swap left/right pairs."""
        kps = kps.copy()
        kps[:, :, 0] = -kps[:, :, 0]  # flip x (normalized coords)

        # Swap pose left-right pairs
        for i, j in self.POSE_FLIP_PAIRS:
            kps[:, i, :], kps[:, j, :] = kps[:, j, :].copy(), kps[:, i, :].copy()

        # Swap left_hand (33:54) ↔ right_hand (54:75)
        tmp = kps[:, 33:54, :].copy()
        kps[:, 33:54, :] = kps[:, 54:75, :]
        kps[:, 54:75, :] = tmp

        return kps

    def _add_noise(self, kps: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, kps.shape).astype(np.float32)
        noise[:, :, 3] = 0  # không noise vào visibility
        return kps + noise

    def _random_scale(self, kps: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        kps = kps.copy()
        kps[:, :, :3] *= scale  # scale x, y, z
        return kps

    def _random_rotation(self, kps: np.ndarray) -> np.ndarray:
        """Rotate around z-axis (in-plane rotation)."""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        kps = kps.copy()
        x = kps[:, :, 0].copy()
        y = kps[:, :, 1].copy()
        kps[:, :, 0] = cos_a * x - sin_a * y
        kps[:, :, 1] = sin_a * x + cos_a * y
        return kps

    def _time_stretch(self, kps: np.ndarray) -> np.ndarray:
        """Thay đổi tốc độ signing bằng linear interpolation."""
        T, N, C = kps.shape
        factor = np.random.uniform(*self.time_stretch_range)
        new_T = max(1, int(T * factor))
        orig_idx = np.linspace(0, T - 1, new_T)
        new_kps = np.zeros((new_T, N, C), dtype=np.float32)
        for i in range(new_T):
            lo = int(np.floor(orig_idx[i]))
            hi = min(lo + 1, T - 1)
            alpha = orig_idx[i] - lo
            new_kps[i] = (1 - alpha) * kps[lo] + alpha * kps[hi]
        return new_kps

    def _frame_dropout(self, kps: np.ndarray) -> np.ndarray:
        """Randomly zero out frames (simulate frame drops)."""
        kps = kps.copy()
        T = kps.shape[0]
        mask = np.random.random(T) < self.frame_drop_prob
        kps[mask] = 0.0
        return kps

    def _joint_dropout(self, kps: np.ndarray) -> np.ndarray:
        """Randomly zero out joints across all frames."""
        kps = kps.copy()
        N = kps.shape[1]
        mask = np.random.random(N) < self.joint_drop_prob
        kps[:, mask, :] = 0.0
        return kps


# ══════════════════════════════════════════════════════════════
#  KEYPOINT NORMALIZER
# ══════════════════════════════════════════════════════════════

class KeypointNormalizer:
    """
    Normalize keypoints relative to body center (hip midpoint).
    Loại bỏ global position, giữ relative motion.
    """

    def __call__(self, kps: np.ndarray) -> np.ndarray:
        """kps: (T, N, 4) — (x, y, z, visibility)"""
        kps = kps.copy().astype(np.float32)

        # Center: midpoint of hips (pose landmarks 23, 24)
        if kps.shape[1] > 24:
            left_hip = kps[:, 23, :3]    # (T, 3)
            right_hip = kps[:, 24, :3]
            center = ((left_hip + right_hip) / 2.0)[:, np.newaxis, :]  # (T, 1, 3)
        else:
            center = kps[:, :, :3].mean(axis=1, keepdims=True)

        kps[:, :, :3] -= center

        # Scale by shoulder width
        if kps.shape[1] > 12:
            ls = kps[:, 11, :3]
            rs = kps[:, 12, :3]
            shoulder_width = np.linalg.norm(ls - rs, axis=-1, keepdims=True)  # (T, 1)
            scale = shoulder_width.mean() + 1e-6
        else:
            scale = (np.abs(kps[:, :, :3]).max() + 1e-6)

        kps[:, :, :3] /= scale
        return kps


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════

# Mapping dataset slug → mode
# "continuous": multi-sign video (dùng CTC loss)
# "isolated":   single-sign video (chỉ dùng CE loss)
DATASET_MODE_MAP = {
    "phoenix":               "continuous",
    "phoenix-2014t":         "continuous",
    "csl":                   "continuous",
    "csl-daily":             "continuous",
    "wlasl":                 "isolated",
    "wlasl-processed":       "isolated",
    "asl-alphabet":          "isolated",
    "sign-language-mnist":   "isolated",
    "kaggle":                "isolated",
    "msasl":                 "isolated",
    "autsl":                 "isolated",
}

def get_dataset_mode(dataset_name: str) -> str:
    """Trả về 'continuous' hoặc 'isolated' từ tên dataset."""
    name_lower = dataset_name.lower().strip()
    for key, mode in DATASET_MODE_MAP.items():
        if key in name_lower:
            return mode
    # Default: nếu gloss có >1 token thì continuous, còn lại isolated
    return "continuous"


class SignLanguageDataset(Dataset):
    """
    Universal Sign Language Dataset.

    Hỗ trợ cả Continuous SL (PHOENIX, CSL) và Isolated SL (WLASL, Kaggle).
    dataset_mode field tự động xác định dựa trên tên dataset:
      - "continuous" → dùng CTC + CE loss
      - "isolated"   → chỉ dùng CE loss (tránh bug CTC với 1 ký hiệu)

    Format JSON mỗi sample:
    {
        "id": "sample_0001",
        "keypoints_path": "data/keypoints/sample_0001.npy",  # (T, 543, 4)
        "gloss": "WEATHER SUNNY TOMORROW",
        "text": "Das Wetter ist morgen sonnig.",
        "dataset": "phoenix",   # phoenix | wlasl | csl | kaggle
        "split": "train"
    }
    
    Hoặc inline:
    {
        "keypoints": [[...], ...],  # nếu nhỏ, lưu thẳng trong JSON
        ...
    }
    """

    def __init__(
        self,
        data_path: str,
        gloss_vocab,
        text_vocab,
        config,
        split: str = "train",
        augment: bool = False,
        max_seq_len: int = 512,
    ):
        self.data_path = Path(data_path)
        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab
        self.config = config
        self.split = split
        self.augment = augment
        self.max_seq_len = max_seq_len

        self.normalizer = KeypointNormalizer()
        self.augmentor = KeypointAugmentor(
            flip_prob=config.aug_flip_prob if augment else 0.0,
            noise_std=config.aug_noise_std if augment else 0.0,
            scale_range=config.aug_scale_range if augment else (1.0, 1.0),
            rotation_range=config.aug_rotation if augment else 0.0,
            time_stretch_range=config.aug_time_stretch if augment else (1.0, 1.0),
            frame_drop_prob=config.aug_frame_drop if augment else 0.0,
            joint_drop_prob=config.aug_joint_drop if augment else 0.0,
        )

        self.samples = self._load_samples()
        print(f"[{split}] Loaded {len(self.samples)} samples from {data_path}")

    def _load_samples(self) -> List[Dict]:
        """Load và validate samples."""
        if self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                samples = json.load(f)
        elif self.data_path.suffix == ".jsonl":
            samples = []
            with open(self.data_path) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")

        # Filter by split nếu cần
        if "split" in samples[0]:
            samples = [s for s in samples if s.get("split", self.split) == self.split]

        return samples

    def _load_keypoints(self, sample: dict) -> np.ndarray:
        """Load keypoints từ file hoặc inline."""
        if "keypoints" in sample:
            kps = np.array(sample["keypoints"], dtype=np.float32)
        elif "keypoints_path" in sample:
            kps = np.load(sample["keypoints_path"]).astype(np.float32)
        else:
            raise ValueError(f"Sample {sample.get('id')} has no keypoints!")

        # Ensure shape (T, N, 4)
        if kps.ndim == 2:
            # (T, N*4) → (T, N, 4)
            T = kps.shape[0]
            kps = kps.reshape(T, -1, 4)
        elif kps.ndim == 3 and kps.shape[-1] == 3:
            # (T, N, 3) → pad visibility column
            T, N, _ = kps.shape
            vis = np.ones((T, N, 1), dtype=np.float32)
            kps = np.concatenate([kps, vis], axis=-1)

        return kps

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load keypoints
        kps = self._load_keypoints(sample)                  # (T, N, 4)

        # Normalize
        kps = self.normalizer(kps)

        # Augment (chỉ train)
        if self.augment:
            kps = self.augmentor(kps)

        # Truncate nếu quá dài
        T = kps.shape[0]
        if T > self.max_seq_len:
            # Random crop khi train, center crop khi val/test
            if self.augment:
                start = random.randint(0, T - self.max_seq_len)
            else:
                start = (T - self.max_seq_len) // 2
            kps = kps[start:start + self.max_seq_len]

        # Pad keypoints đến 543 nếu ít hơn (một số dataset ít joints hơn)
        N = kps.shape[1]
        if N < 543:
            pad = np.zeros((kps.shape[0], 543 - N, 4), dtype=np.float32)
            kps = np.concatenate([kps, pad], axis=1)

        kps_tensor = torch.from_numpy(kps)                  # (T, 543, 4)

        # Gloss encoding
        gloss_str = sample.get("gloss", "")
        gloss_ids = self.gloss_vocab.encode(gloss_str)
        gloss_tensor = torch.tensor(gloss_ids, dtype=torch.long)

        # Text encoding
        text_str = sample.get("text", "")
        text_ids = self.text_vocab.encode(text_str, add_bos=True, add_eos=True)
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        # Xác định dataset_mode dựa trên tên dataset
        dataset_name = sample.get("dataset", "unknown")
        # Cho phép sample tự khai báo mode (override)
        if "dataset_mode" in sample:
            mode = sample["dataset_mode"]
        else:
            # Auto-detect: isolated nếu chỉ có 1 gloss token
            detected = get_dataset_mode(dataset_name)
            if detected == "continuous" and len(gloss_ids) <= 1:
                detected = "isolated"  # Single-sign → dùng CE only
            mode = detected

        return {
            "id": sample.get("id", str(idx)),
            "keypoints": kps_tensor,                        # (T, 543, 4)
            "gloss_ids": gloss_tensor,                      # (Lg,)
            "text_ids": text_tensor,                        # (Lw,)
            "dataset": dataset_name,
            "dataset_mode": mode,                           # "continuous" | "isolated"
        }


# ══════════════════════════════════════════════════════════════
#  COLLATE
# ══════════════════════════════════════════════════════════════

def collate_fn(batch: List[dict]) -> dict:
    """
    Dynamic padding cho variable-length sequences.
    Giữ nguyên dataset_mode (list of strings, không cần pad).
    """
    ids = [b["id"] for b in batch]
    keypoints = [b["keypoints"] for b in batch]             # list of (T_i, 543, 4)
    gloss_ids = [b["gloss_ids"] for b in batch]
    text_ids = [b["text_ids"] for b in batch]
    dataset_modes = [b.get("dataset_mode", "continuous") for b in batch]

    # Actual lengths
    kp_lengths = torch.tensor([k.shape[0] for k in keypoints], dtype=torch.long)
    gloss_lengths = torch.tensor([g.shape[0] for g in gloss_ids], dtype=torch.long)
    text_lengths = torch.tensor([t.shape[0] for t in text_ids], dtype=torch.long)

    # Pad keypoints: (B, T_max, 543, 4)
    max_T = kp_lengths.max().item()
    B = len(batch)
    kp_padded = torch.zeros(B, max_T, 543, 4)
    for i, kp in enumerate(keypoints):
        T = kp.shape[0]
        kp_padded[i, :T] = kp

    # Pad glosses: (B, Lg_max)
    gloss_padded = pad_sequence(gloss_ids, batch_first=True, padding_value=0)

    # Pad text: (B, Lw_max)
    text_padded = pad_sequence(text_ids, batch_first=True, padding_value=0)

    return {
        "ids": ids,
        "keypoints": kp_padded,
        "keypoint_lengths": kp_lengths,
        "gloss_ids": gloss_padded,
        "gloss_lengths": gloss_lengths,
        "text_ids": text_padded,
        "text_lengths": text_lengths,
        "dataset_mode": dataset_modes,      # list[str]: "continuous" | "isolated"
    }


# ══════════════════════════════════════════════════════════════
#  KAGGLE DATA LOADER (dùng kagglehub)
# ══════════════════════════════════════════════════════════════

def load_kaggle_sign_dataset(
    dataset_handle: str,
    output_json_path: str,
    split_ratio: tuple = (0.8, 0.1, 0.1),
):
    """
    Download dataset từ Kaggle và convert sang format JSON.
    
    Ví dụ:
        load_kaggle_sign_dataset(
            "datamunge/sign-language-mnist",
            "data/kaggle_sign.json"
        )
    
    Args:
        dataset_handle: Kaggle dataset slug (owner/dataset-name)
        output_json_path: Nơi lưu JSON đã convert
        split_ratio: (train, val, test) split ratios
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    print(f"Downloading {dataset_handle} from Kaggle...")

    try:
        # Download to local cache
        path = kagglehub.dataset_download(dataset_handle)
        print(f"Downloaded to: {path}")

        # Try to load as pandas
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            dataset_handle,
        )
        print(f"Loaded DataFrame: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Convert to our JSON format (cần adapt theo dataset cụ thể)
        samples = _convert_dataframe_to_samples(df, dataset_handle)

    except Exception as e:
        print(f"Pandas load failed ({e}), trying file download...")
        samples = _load_from_files(path, dataset_handle)

    # Split
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    for i, s in enumerate(samples[:n_train]):
        s["split"] = "train"
    for i, s in enumerate(samples[n_train:n_train+n_val]):
        s["split"] = "val"
    for i, s in enumerate(samples[n_train+n_val:]):
        s["split"] = "test"

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(samples, f)
    print(f"Saved {len(samples)} samples to {output_json_path}")
    return samples


def _convert_dataframe_to_samples(df, dataset_handle: str) -> list:
    """
    Convert pandas DataFrame sang format chuẩn.
    Cần customize theo từng dataset.
    """
    samples = []
    
    # Detect columns
    cols = list(df.columns)
    
    # Sign Language MNIST format: label + pixel columns
    if "label" in cols and any(c.startswith("pixel") for c in cols):
        pixel_cols = [c for c in cols if c.startswith("pixel")]
        for i, row in df.iterrows():
            label = int(row["label"])
            pixels = row[pixel_cols].values.astype(np.float32)
            # MNIST: 28x28 image → dùng như 1 frame với fake keypoints
            # (trong thực tế nên extract keypoints từ image)
            fake_kps = np.zeros((1, 543, 4), dtype=np.float32)
            # Store as inline (nhỏ)
            samples.append({
                "id": f"kaggle_{i}",
                "keypoints": fake_kps.tolist(),
                "gloss": str(label),
                "text": str(label),
                "dataset": dataset_handle,
            })
    
    # Generic: nếu có cột keypoints
    elif any("keypoint" in c.lower() for c in cols):
        kp_cols = [c for c in cols if "keypoint" in c.lower()]
        for i, row in df.iterrows():
            samples.append({
                "id": f"kaggle_{i}",
                "keypoints": [[row[c] for c in kp_cols]],
                "gloss": str(row.get("gloss", row.get("label", "unknown"))),
                "text": str(row.get("text", row.get("translation", ""))),
                "dataset": dataset_handle,
            })
    
    else:
        print(f"Warning: Unknown DataFrame format for {dataset_handle}")
        print(f"Columns: {cols[:10]}")
    
    return samples


def _load_from_files(path: str, dataset_handle: str) -> list:
    """Load từ files đã download (video hoặc npy)."""
    import glob
    samples = []
    path = Path(path)
    
    # Tìm npy files (pre-extracted keypoints)
    npy_files = list(path.rglob("*.npy"))
    if npy_files:
        for f in npy_files:
            samples.append({
                "id": f.stem,
                "keypoints_path": str(f),
                "gloss": f.stem.split("_")[0],  # lấy từ filename
                "text": "",
                "dataset": dataset_handle,
            })
    
    # Tìm JSON annotation files
    json_files = list(path.rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf) as fp:
                data = json.load(fp)
            if isinstance(data, list):
                samples.extend(data)
        except Exception:
            pass
    
    return samples


# ══════════════════════════════════════════════════════════════
#  PREPROCESS: VIDEO → KEYPOINTS (dùng MediaPipe)
# ══════════════════════════════════════════════════════════════

def extract_keypoints_from_video(
    video_path: str,
    output_path: str,
    target_fps: int = 25,
) -> Optional[np.ndarray]:
    """
    Extract MediaPipe Holistic keypoints từ video.
    Lưu dưới dạng .npy (T, 543, 4).
    
    Cần: pip install mediapipe opencv-python
    """
    try:
        import cv2
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open: {video_path}")
            return None

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_skip = max(1, int(orig_fps / target_fps))

        all_keypoints = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                kps = _extract_landmarks(results)  # (543, 4)
                all_keypoints.append(kps)

            frame_idx += 1

        cap.release()
        holistic.close()

        if not all_keypoints:
            return None

        keypoints = np.stack(all_keypoints, axis=0).astype(np.float32)  # (T, 543, 4)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, keypoints)

        return keypoints

    except ImportError:
        print("mediapipe/opencv not installed. Run: pip install mediapipe opencv-python")
        return None


def _extract_landmarks(results) -> np.ndarray:
    """Extract 543 landmarks từ MediaPipe Holistic results."""
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic

    kps = np.zeros((543, 4), dtype=np.float32)

    def fill(landmarks, start, n):
        if landmarks:
            for i, lm in enumerate(landmarks.landmark[:n]):
                kps[start + i] = [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]

    fill(results.pose_landmarks, 0, 33)
    fill(results.left_hand_landmarks, 33, 21)
    fill(results.right_hand_landmarks, 54, 21)
    fill(results.face_landmarks, 75, 468)

    return kps


def batch_extract_keypoints(
    video_dir: str,
    output_dir: str,
    annotation_file: str,
    num_workers: int = 4,
):
    """
    Batch extract keypoints từ toàn bộ video directory.
    Dùng multiprocessing để nhanh hơn.
    """
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(video_dir.rglob("*.mp4")) + \
                  list(video_dir.rglob("*.avi")) + \
                  list(video_dir.rglob("*.mov"))

    print(f"Found {len(video_files)} videos in {video_dir}")

    def process_one(vf):
        out = output_dir / (vf.stem + ".npy")
        if out.exists():
            return str(out)
        kps = extract_keypoints_from_video(str(vf), str(out))
        return str(out) if kps is not None else None

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for r in tqdm(executor.map(process_one, video_files), total=len(video_files)):
            if r:
                results.append(r)

    print(f"Successfully extracted {len(results)}/{len(video_files)} videos")
    return results
