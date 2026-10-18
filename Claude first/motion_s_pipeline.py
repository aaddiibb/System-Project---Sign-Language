# =============================================================================
# Motion-S: Text-to-Sign Motion Generation — Full Kaggle Pipeline
# Competition: https://kaggle.com/competitions/motion-s-hierarchical-text-to-motion-generation-for-sign-language
#
# STRATEGY: Three-tier hybrid approach
#   Tier 1 — Semantic Retrieval  (fast, zero-shot, high coverage)
#   Tier 2 — Progressive Transformer  (trained seq2seq, better generalisation)
#   Tier 3 — Diffusion Refinement  (optional, improves realism on GPU)
#   Post-proc — Cosine smoothing + temporal normalisation
#   Eval     — DTW-MJE + BLEU-4 back-translation
# =============================================================================

# ── 0. INSTALL (run once in Kaggle notebook) ──────────────────────────────────
# !pip install -q sentence-transformers dtw-python sacrebleu einops timm

# =============================================================================
# SECTION 1 — IMPORTS & CONFIG
# =============================================================================
import os, json, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

from dtw import dtw as dtw_fn          # pip install dtw-python
import sacrebleu                        # pip install sacrebleu

warnings.filterwarnings("ignore")

# ── Paths (Kaggle layout) ────────────────────────────────────────────────────
INPUT   = Path("/kaggle/input/motion-s-hierarchical-text-to-motion-generation-for-sign-language")
WORKING = Path("/kaggle/working")
WORKING.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    max_seq_len    = 200,       # max pose frames per sentence
    num_joints     = 128,       # DWPose: 18 body + 21 lh + 21 rh + 68 face
    joint_dim      = 3,         # x, y, conf  (SignVerse-2M format)
    pose_dim       = 128 * 3,   # flattened per frame  →  384

    # Retrieval
    top_k          = 5,         # kNN neighbours for retrieval

    # Transformer
    d_model        = 256,
    nhead          = 8,
    num_enc_layers = 4,
    num_dec_layers = 4,
    dim_ff         = 1024,
    dropout        = 0.1,
    max_text_len   = 128,       # BPE tokens

    # Training
    epochs         = 60,
    batch_size     = 16,
    lr             = 1e-4,
    weight_decay   = 1e-4,
    warmup_epochs  = 5,
    grad_clip      = 1.0,

    # Diffusion (optional refinement)
    diff_steps     = 100,
    diff_beta_min  = 1e-4,
    diff_beta_max  = 2e-2,

    # Post-processing
    smooth_window  = 5,         # cosine smoothing frames

    device         = "cuda" if torch.cuda.is_available() else "cpu",
    seed           = 42,
)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"[Config] device={CFG['device']}  pose_dim={CFG['pose_dim']}")


# =============================================================================
# SECTION 2 — DATA LOADING
# =============================================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expected layout inside the competition input folder:
        train.csv      —  columns: id, text, num_frames
        test.csv       —  columns: id, text
        train_poses/   —  {id}.npy  shape (T, 128, 3) each
        (test poses are what we must generate)
    Adjust column names below if the actual CSV differs.
    """
    train_df = pd.read_csv(INPUT / "train.csv")
    test_df  = pd.read_csv(INPUT / "test.csv")

    print(f"[Data] train={len(train_df)}  test={len(test_df)}")
    print(f"[Data] train columns: {train_df.columns.tolist()}")
    return train_df, test_df


def load_pose(sample_id: str, split: str = "train") -> Optional[np.ndarray]:
    """Load a single pose .npy file. Returns (T, 128, 3) or None."""
    folder = INPUT / f"{split}_poses"
    path   = folder / f"{sample_id}.npy"
    if not path.exists():
        return None
    pose = np.load(str(path), allow_pickle=True)
    if pose.ndim == 2:                        # (T, 384) → (T, 128, 3)
        pose = pose.reshape(-1, 128, 3)
    return pose.astype(np.float32)


def pad_or_trim(pose: np.ndarray, max_len: int) -> np.ndarray:
    """Pad or trim pose sequence to max_len frames."""
    T = pose.shape[0]
    if T >= max_len:
        return pose[:max_len]
    pad = np.zeros((max_len - T, pose.shape[1], pose.shape[2]), dtype=np.float32)
    return np.concatenate([pose, pad], axis=0)


# =============================================================================
# SECTION 3 — TEXT PREPROCESSING
# =============================================================================
def preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, normalise whitespace."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 '-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# SECTION 4 — TIER 1: SEMANTIC RETRIEVAL (TF-IDF + Sentence-BERT + kNN)
#
# Best for: test sentences whose vocabulary overlaps heavily with train.
# Inspired by the public TF-IDF+kNN notebook on the competition page.
# Upgraded: dual-encoder (TF-IDF + SBERT) for better semantic coverage.
# =============================================================================
class SemanticRetriever:
    """
    Dual-encoder retrieval:
      1. TF-IDF sparse bag-of-words (handles exact matches / rare words)
      2. Sentence-BERT dense embeddings (handles paraphrase / synonyms)
      Final score = alpha * tfidf_sim + (1-alpha) * sbert_sim
    """
    def __init__(self, alpha: float = 0.4):
        self.alpha     = alpha
        self.tfidf     = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)
        self.sbert     = SentenceTransformer("all-MiniLM-L6-v2")
        self.knn_tfidf = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn_sbert = NearestNeighbors(metric="cosine", algorithm="brute")
        self.train_ids : List[str] = []

    def fit(self, train_df: pd.DataFrame):
        texts = train_df["text"].apply(preprocess_text).tolist()
        self.train_ids = train_df["id"].tolist()

        # TF-IDF
        tfidf_vecs = self.tfidf.fit_transform(texts)
        self.knn_tfidf.fit(tfidf_vecs)
        self.tfidf_vecs = tfidf_vecs

        # Sentence-BERT
        print("[Retriever] encoding training texts with SBERT …")
        sbert_vecs = self.sbert.encode(texts, batch_size=64,
                                        show_progress_bar=True,
                                        normalize_embeddings=True)
        self.knn_sbert.fit(sbert_vecs)
        self.sbert_vecs = sbert_vecs
        print(f"[Retriever] fitted on {len(texts)} training examples.")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return [(train_id, score), …] sorted best-first."""
        q = preprocess_text(query)

        # TF-IDF distances
        q_tfidf = self.tfidf.transform([q])
        d_tfidf, idx_tfidf = self.knn_tfidf.kneighbors(q_tfidf, n_neighbors=k)
        sim_tfidf = 1 - d_tfidf[0]          # cosine similarity

        # SBERT distances
        q_sbert = self.sbert.encode([q], normalize_embeddings=True)
        d_sbert, idx_sbert = self.knn_sbert.kneighbors(q_sbert, n_neighbors=k)
        sim_sbert = 1 - d_sbert[0]

        # Combine: union of top-k from both encoders, score blended
        combined: Dict[int, float] = {}
        for i, s in zip(idx_tfidf[0], sim_tfidf):
            combined[i] = combined.get(i, 0) + self.alpha * s
        for i, s in zip(idx_sbert[0], sim_sbert):
            combined[i] = combined.get(i, 0) + (1 - self.alpha) * s

        ranked = sorted(combined.items(), key=lambda x: -x[1])[:k]
        return [(self.train_ids[i], score) for i, score in ranked]

    def predict_pose(self, query: str, k: int = 5) -> np.ndarray:
        """
        Retrieve top-k poses and blend them by similarity score.
        Returns (max_seq_len, 128, 3).
        """
        hits = self.retrieve(query, k=k)
        poses, weights = [], []
        for tid, score in hits:
            pose = load_pose(tid, split="train")
            if pose is not None:
                poses.append(pad_or_trim(pose, CFG["max_seq_len"]))
                weights.append(score)

        if not poses:
            return np.zeros((CFG["max_seq_len"], 128, 3), dtype=np.float32)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()
        blended = sum(w * p for w, p in zip(weights, poses))
        return blended.astype(np.float32)


# =============================================================================
# SECTION 5 — TIER 2: PROGRESSIVE TRANSFORMER (Text → Pose seq2seq)
#
# Architecture:
#   Encoder : word-embedding + positional encoding → Transformer encoder
#   Decoder : autoregressive Transformer decoder
#             at each step predicts next pose frame  (384-d vector)
#   Loss    : L1 reconstruction + velocity loss (smooth motion)
# Reference: Saunders et al. ECCV 2020 (Progressive Transformers for SLP)
# Enhancement: added velocity loss + label smoothing from SLRTP2025 winners
# =============================================================================

# ── Tokeniser (simple BPE via HuggingFace tokenizers) ──────────────────────
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
VOCAB_SIZE = TOKENIZER.vocab_size


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class ProgressiveTransformer(nn.Module):
    """
    Text → continuous pose sequence.
    Input:  token ids  (B, L)
    Output: pose frames (B, T, pose_dim)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        D  = cfg["d_model"]
        PD = cfg["pose_dim"]

        # Encoder
        self.src_embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=0)
        self.src_pe    = PositionalEncoding(D, cfg["max_text_len"], cfg["dropout"])
        enc_layer      = nn.TransformerEncoderLayer(
            D, cfg["nhead"], cfg["dim_ff"],
            cfg["dropout"], batch_first=True, norm_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, cfg["num_enc_layers"])

        # Decoder
        self.tgt_proj  = nn.Linear(PD, D)
        self.tgt_pe    = PositionalEncoding(D, cfg["max_seq_len"] + 1, cfg["dropout"])
        dec_layer      = nn.TransformerDecoderLayer(
            D, cfg["nhead"], cfg["dim_ff"],
            cfg["dropout"], batch_first=True, norm_first=True)
        self.decoder   = nn.TransformerDecoder(dec_layer, cfg["num_dec_layers"])

        # Output head: pose prediction + stop token
        self.out_pose  = nn.Linear(D, PD)
        self.out_stop  = nn.Linear(D, 1)    # binary: has sequence ended?

        # Counter embedding for progressive decoding
        self.counter   = nn.Embedding(cfg["max_seq_len"] + 1, D)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids: torch.Tensor,
               src_mask: torch.Tensor) -> torch.Tensor:
        emb = self.src_pe(self.src_embed(src_ids))
        return self.encoder(emb, src_key_padding_mask=(src_ids == 0))

    def decode_step(self, tgt: torch.Tensor,
                    memory: torch.Tensor,
                    src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single decode step. tgt: (B, T, pose_dim)"""
        B, T, _ = tgt.shape
        steps = torch.arange(T, device=tgt.device).unsqueeze(0).expand(B, -1)
        tgt_d = self.tgt_pe(self.tgt_proj(tgt) + self.counter(steps))

        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)
        out = self.decoder(tgt_d, memory,
                           tgt_mask=causal,
                           memory_key_padding_mask=(src_mask == 0))
        return self.out_pose(out), self.out_stop(out).squeeze(-1)

    def forward(self, src_ids: torch.Tensor,
                tgt_poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src_ids   : (B, L)   text token ids
        tgt_poses : (B, T, pose_dim)   teacher-forced targets
        Returns   : pred_poses (B, T, pose_dim),  stop_logits (B, T)
        """
        src_mask = (src_ids != 0).float()
        memory   = self.encode(src_ids, src_mask)
        # Prepend zero-frame (BOS) for teacher forcing
        bos = torch.zeros(tgt_poses.size(0), 1, tgt_poses.size(2),
                          device=tgt_poses.device)
        tgt_input = torch.cat([bos, tgt_poses[:, :-1]], dim=1)
        return self.decode_step(tgt_input, memory, src_mask)

    @torch.no_grad()
    def generate(self, src_ids: torch.Tensor,
                 max_len: int = 200,
                 stop_threshold: float = 0.5) -> np.ndarray:
        """Autoregressive generation at inference time."""
        self.eval()
        src_mask = (src_ids != 0).float()
        memory   = self.encode(src_ids, src_mask)
        B        = src_ids.size(0)
        PD       = CFG["pose_dim"]

        generated = torch.zeros(B, 1, PD, device=src_ids.device)
        for _ in range(max_len - 1):
            preds, stops = self.decode_step(generated, memory, src_mask)
            next_frame   = preds[:, -1:, :]   # (B, 1, PD)
            generated    = torch.cat([generated, next_frame], dim=1)
            # Early stop: all batch items issued stop token
            if (torch.sigmoid(stops[:, -1]) > stop_threshold).all():
                break

        # Remove BOS frame, reshape to (B, T, 128, 3)
        out = generated[:, 1:].cpu().numpy()
        return out.reshape(B, -1, 128, 3)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SignDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str = "train"):
        self.df    = df.reset_index(drop=True)
        self.split = split
        self.max_L = CFG["max_seq_len"]
        self.PD    = CFG["pose_dim"]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        text = preprocess_text(str(row["text"]))

        # Tokenise
        enc = TOKENIZER(
            text,
            max_length      = CFG["max_text_len"],
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        src_ids = enc["input_ids"].squeeze(0)   # (L,)

        if self.split == "train":
            pose = load_pose(str(row["id"]), split="train")
            if pose is None:
                pose = np.zeros((self.max_L, 128, 3), dtype=np.float32)
            pose  = pad_or_trim(pose, self.max_L)   # (T, 128, 3)
            # Flatten joints: (T, 384)
            pose_flat = pose.reshape(self.max_L, -1)
            return src_ids, torch.tensor(pose_flat, dtype=torch.float32)
        else:
            return src_ids, row["id"]


def collate_fn(batch):
    if isinstance(batch[0][1], torch.Tensor):
        src = torch.stack([b[0] for b in batch])
        tgt = torch.stack([b[1] for b in batch])
        return src, tgt
    else:
        src = torch.stack([b[0] for b in batch])
        ids = [b[1] for b in batch]
        return src, ids


# ── Losses ────────────────────────────────────────────────────────────────────
class SLPLoss(nn.Module):
    """
    L_total = L1_reconstruction + λ_vel * velocity_loss + λ_stop * BCE_stop
    Velocity loss: penalise jerky motion  →  encourages smooth transitions
    """
    def __init__(self, lambda_vel: float = 0.5, lambda_stop: float = 0.1):
        super().__init__()
        self.lv   = lambda_vel
        self.ls   = lambda_stop
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred_pose: torch.Tensor,
                tgt_pose: torch.Tensor,
                stop_logits: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        # Reconstruction (L1)
        l_rec = F.l1_loss(pred_pose * tgt_mask.unsqueeze(-1),
                           tgt_pose  * tgt_mask.unsqueeze(-1))

        # Velocity (first-order difference)
        pred_vel = pred_pose[:, 1:] - pred_pose[:, :-1]
        tgt_vel  = tgt_pose[:, 1:]  - tgt_pose[:, :-1]
        l_vel    = F.l1_loss(pred_vel, tgt_vel)

        # Stop token (1 at last real frame, 0 otherwise)
        T        = tgt_mask.size(1)
        stop_tgt = torch.zeros_like(stop_logits)
        for b in range(tgt_mask.size(0)):
            # last valid frame index
            last = tgt_mask[b].sum().long() - 1
            if last >= 0:
                stop_tgt[b, last] = 1.0
        l_stop   = self.bce(stop_logits, stop_tgt)

        return l_rec + self.lv * l_vel + self.ls * l_stop


# ── Trainer ───────────────────────────────────────────────────────────────────
def train_transformer(train_df: pd.DataFrame,
                      val_df: pd.DataFrame) -> ProgressiveTransformer:
    model = ProgressiveTransformer(CFG).to(CFG["device"])
    opt   = AdamW(model.parameters(), lr=CFG["lr"],
                  weight_decay=CFG["weight_decay"])
    sch   = CosineAnnealingLR(opt, T_max=CFG["epochs"],
                               eta_min=CFG["lr"] * 0.1)
    crit  = SLPLoss()

    train_ds = SignDataset(train_df, "train")
    val_ds   = SignDataset(val_df,   "train")   # val also has GT
    train_ld = DataLoader(train_ds, CFG["batch_size"], shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds, CFG["batch_size"], shuffle=False,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)

    scaler   = torch.cuda.amp.GradScaler(enabled=(CFG["device"] == "cuda"))
    best_val = float("inf")

    for epoch in range(1, CFG["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for src, tgt in train_ld:
            src, tgt = src.to(CFG["device"]), tgt.to(CFG["device"])
            # Build mask: non-zero frames
            tgt_mask = (tgt.abs().sum(-1) > 1e-6).float()

            with torch.cuda.amp.autocast(enabled=(CFG["device"] == "cuda")):
                pred, stops = model(src, tgt)
                loss        = crit(pred, tgt, stops, tgt_mask)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        sch.step()
        avg_train = total_loss / len(train_ld)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_ld:
                src, tgt = src.to(CFG["device"]), tgt.to(CFG["device"])
                tgt_mask = (tgt.abs().sum(-1) > 1e-6).float()
                with torch.cuda.amp.autocast(enabled=(CFG["device"] == "cuda")):
                    pred, stops = model(src, tgt)
                    val_loss   += crit(pred, tgt, stops, tgt_mask).item()
        avg_val = val_loss / max(len(val_ld), 1)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), WORKING / "best_transformer.pt")
            print(f"  ✓ saved checkpoint  val={avg_val:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{CFG['epochs']} | "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={sch.get_last_lr()[0]:.2e}")

    # Load best
    model.load_state_dict(torch.load(WORKING / "best_transformer.pt",
                                     map_location=CFG["device"]))
    return model


# =============================================================================
# SECTION 6 — TIER 3: LIGHTWEIGHT DIFFUSION REFINER  (optional, GPU required)
#
# Operates in the same pose space as the Transformer output.
# Takes the Transformer's noisy prediction and denoises it.
# Inspired by Sign-IDD (AAAI 2025) and G2P-DDM (AAAI 2024).
# =============================================================================
class SinusoidalTimestepEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freq  = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args  = t[:, None].float() * freq[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionRefiner(nn.Module):
    """
    Denoising model: given noisy pose sequence + timestep + text memory,
    predict the noise (ε-prediction a la DDPM).
    Architecture: lightweight 1-D convolutional UNet with cross-attention to text.
    """
    def __init__(self, pose_dim: int = 384, d_model: int = 256,
                 n_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbed(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.in_proj = nn.Linear(pose_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, d_model * 4, 0.1,
            batch_first=True, norm_first=True)
        self.net   = nn.TransformerDecoder(dec_layer, n_layers)
        self.out   = nn.Linear(d_model, pose_dim)

    def forward(self, x_noisy: torch.Tensor,
                t: torch.Tensor,
                text_memory: torch.Tensor) -> torch.Tensor:
        B, T, _   = x_noisy.shape
        t_emb     = self.time_embed(t)           # (B, d_model)
        h         = self.in_proj(x_noisy)        # (B, T, d_model)
        h         = h + t_emb.unsqueeze(1)       # broadcast over T
        noise_pred = self.out(self.net(h, text_memory))
        return noise_pred


class GaussianDiffusion:
    """DDPM noise schedule + reverse diffusion."""
    def __init__(self, steps: int = 100,
                 beta_min: float = 1e-4, beta_max: float = 2e-2):
        self.T      = steps
        betas       = torch.linspace(beta_min, beta_max, steps)
        alphas      = 1 - betas
        alpha_bar   = torch.cumprod(alphas, dim=0)
        self.betas      = betas
        self.alpha_bar  = alpha_bar
        self.sqrt_ab    = alpha_bar.sqrt()
        self.sqrt_1m_ab = (1 - alpha_bar).sqrt()

    def q_sample(self, x0: torch.Tensor,
                 t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.sqrt_ab[t].to(x0.device)
        mab = self.sqrt_1m_ab[t].to(x0.device)
        return ab[:, None, None] * x0 + mab[:, None, None] * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model: DiffusionRefiner,
                      x_init: torch.Tensor,
                      text_memory: torch.Tensor,
                      start_t: int = 50) -> torch.Tensor:
        """
        DDIM-style reverse from start_t → 0.
        x_init: coarse pose from Transformer  (B, T, pose_dim)
        """
        device = x_init.device
        x = x_init.clone()

        # Add a mild amount of noise at start_t
        t_tensor = torch.full((x.size(0),), start_t - 1,
                              dtype=torch.long, device=device)
        x, _ = self.q_sample(x, t_tensor)

        for t_val in range(start_t - 1, -1, -1):
            t_tensor = torch.full((x.size(0),), t_val,
                                  dtype=torch.long, device=device)
            eps_pred = model(x, t_tensor, text_memory)
            ab  = self.sqrt_ab[t_val].to(device)
            mab = self.sqrt_1m_ab[t_val].to(device)
            x0_pred = (x - mab * eps_pred) / ab
            if t_val > 0:
                beta  = self.betas[t_val].to(device)
                noise = torch.randn_like(x)
                x = (1 - beta).sqrt() * x0_pred + beta.sqrt() * noise
            else:
                x = x0_pred
        return x


def train_diffusion(train_df: pd.DataFrame,
                    transformer: ProgressiveTransformer) -> DiffusionRefiner:
    """
    Train the diffusion refiner on (noisy_pose → clean_pose) pairs.
    Uses the Transformer encoder as frozen text memory provider.
    """
    diff_model = DiffusionRefiner(
        pose_dim = CFG["pose_dim"],
        d_model  = CFG["d_model"],
    ).to(CFG["device"])

    diffusion  = GaussianDiffusion(CFG["diff_steps"],
                                   CFG["diff_beta_min"],
                                   CFG["diff_beta_max"])
    opt        = AdamW(diff_model.parameters(), lr=1e-4)
    scaler     = torch.cuda.amp.GradScaler(enabled=(CFG["device"] == "cuda"))
    ds         = SignDataset(train_df, "train")
    ld         = DataLoader(ds, 8, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)

    diff_epochs = 30
    transformer.eval()
    for epoch in range(1, diff_epochs + 1):
        total = 0.0
        for src, tgt in ld:
            src, tgt = src.to(CFG["device"]), tgt.to(CFG["device"])
            B = src.size(0)
            t = torch.randint(0, CFG["diff_steps"], (B,), device=CFG["device"])

            # Get text memory (frozen encoder)
            with torch.no_grad():
                mem = transformer.encode(src, (src != 0).float())

            x_noisy, noise = diffusion.q_sample(tgt, t)

            with torch.cuda.amp.autocast(enabled=(CFG["device"] == "cuda")):
                eps_pred = diff_model(x_noisy, t, mem)
                loss     = F.mse_loss(eps_pred, noise)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item()

        if epoch % 5 == 0:
            print(f"  [Diffusion] epoch {epoch}/{diff_epochs}  "
                  f"loss={total/len(ld):.4f}")

    torch.save(diff_model.state_dict(), WORKING / "diff_refiner.pt")
    return diff_model


# =============================================================================
# SECTION 7 — POST-PROCESSING
# =============================================================================
def cosine_smooth(pose: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Cosine-weighted temporal smoothing to reduce jitter between frames.
    pose: (T, 128, 3)
    """
    T = pose.shape[0]
    if T <= window:
        return pose
    w = np.hanning(window)
    w = w / w.sum()
    out = pose.copy()
    half = window // 2
    for t in range(half, T - half):
        out[t] = sum(w[i] * pose[t - half + i] for i in range(window))
    return out.astype(np.float32)


def normalise_pose(pose: np.ndarray) -> np.ndarray:
    """
    Root-centre the skeleton (subtract hip/body centroid) and
    scale so the torso height = 1.  Makes poses signer-agnostic.
    pose: (T, 128, 3)
    """
    # Body joints are indices 0-17 in DWPose (18 body joints)
    body = pose[:, :18, :]
    # Hip centre ≈ mean of joints 11 and 12 (left/right hip in DWPose)
    hip = body[:, [11, 12], :2].mean(axis=1, keepdims=True)   # (T,1,2)
    pose = pose.copy()
    pose[:, :, :2] -= hip

    # Scale: shoulder width (joints 5,6) as reference
    shoulder_w = np.linalg.norm(
        body[:, 5, :2] - body[:, 6, :2], axis=-1
    ).mean()
    if shoulder_w > 1e-6:
        pose[:, :, :2] /= shoulder_w
    return pose.astype(np.float32)


# =============================================================================
# SECTION 8 — EVALUATION METRICS
# =============================================================================
def dtw_mje(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    DTW-MJE: Dynamic Time Warping + Mean Joint Error.
    Lower is better.
    pred, gt: (T, 128, 3) or (T, 384)
    """
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
    if gt.ndim == 3:
        gt   = gt.reshape(gt.shape[0], -1)
    alignment = dtw_fn(pred, gt, distance_only=False)
    path      = np.array(alignment.path)
    errors    = [np.mean(np.abs(pred[i] - gt[j])) for i, j in path]
    return float(np.mean(errors))


def compute_bleu_backtranslation(pred_poses: List[np.ndarray],
                                  ref_texts: List[str],
                                  back_translator) -> float:
    """
    Back-translate generated poses → text, then BLEU vs. original text.
    Requires a trained Sign-to-Text model (back_translator).
    Here we provide the interface; plug in your trained S2T model.
    """
    hyps, refs = [], []
    for pose, ref in zip(pred_poses, ref_texts):
        try:
            hyp = back_translator(pose)   # your S2T model
        except Exception:
            hyp = ""
        hyps.append(hyp)
        refs.append([ref])

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score


def evaluate_on_val(model: ProgressiveTransformer,
                    val_df: pd.DataFrame,
                    n_samples: int = 100) -> Dict[str, float]:
    """Compute DTW-MJE on a validation subset."""
    sample = val_df.sample(min(n_samples, len(val_df)), random_state=42)
    dtw_scores = []

    for _, row in sample.iterrows():
        gt = load_pose(str(row["id"]), split="train")
        if gt is None:
            continue

        # Tokenise
        enc = TOKENIZER(
            preprocess_text(str(row["text"])),
            max_length=CFG["max_text_len"], padding="max_length",
            truncation=True, return_tensors="pt"
        )
        src = enc["input_ids"].to(CFG["device"])
        pred_4d = model.generate(src, max_len=gt.shape[0] + 20)
        pred    = pred_4d[0]  # (T, 128, 3)
        pred    = pred[:gt.shape[0]]  # align lengths

        score = dtw_mje(pred, gt)
        dtw_scores.append(score)

    mean_dtw = float(np.mean(dtw_scores)) if dtw_scores else 0.0
    print(f"[Eval] DTW-MJE on {len(dtw_scores)} samples: {mean_dtw:.4f}")
    return {"dtw_mje": mean_dtw}


# =============================================================================
# SECTION 9 — HYBRID INFERENCE PIPELINE
# =============================================================================
def hybrid_predict(
    text: str,
    retriever: SemanticRetriever,
    transformer: ProgressiveTransformer,
    diffusion_refiner: Optional[DiffusionRefiner] = None,
    diffusion_obj: Optional[GaussianDiffusion] = None,
    retrieval_threshold: float = 0.75,
) -> np.ndarray:
    """
    Decision logic:
      1. Retrieve top-1 match; if score ≥ threshold → use retrieval (exact/near-exact)
      2. Otherwise → Transformer generation
      3. (Optional) Refine with diffusion
      4. Post-process: normalise + smooth
    Returns: (T, 128, 3)
    """
    # ── Retrieval ─────────────────────────────────────────────────────────────
    hits = retriever.retrieve(text, k=CFG["top_k"])
    top_score = hits[0][1] if hits else 0.0

    if top_score >= retrieval_threshold:
        # High-confidence retrieval: blend top-k
        pose = retriever.predict_pose(text, k=CFG["top_k"])
    else:
        # Low-confidence: generate with Transformer
        enc = TOKENIZER(
            preprocess_text(text),
            max_length=CFG["max_text_len"], padding="max_length",
            truncation=True, return_tensors="pt"
        )
        src = enc["input_ids"].to(CFG["device"])
        pred_4d = transformer.generate(src, max_len=CFG["max_seq_len"])
        pose    = pred_4d[0]   # (T, 128, 3)

        # ── Optionally blend retrieval + generation ──────────────────────────
        if top_score > 0.4:
            retr_pose = retriever.predict_pose(text, k=3)
            alpha = (top_score - 0.4) / (retrieval_threshold - 0.4)  # 0→1
            pose = (1 - alpha) * pose + alpha * retr_pose

    # ── Diffusion refinement ─────────────────────────────────────────────────
    if diffusion_refiner is not None and diffusion_obj is not None:
        enc = TOKENIZER(
            preprocess_text(text),
            max_length=CFG["max_text_len"], padding="max_length",
            truncation=True, return_tensors="pt"
        )
        src = enc["input_ids"].to(CFG["device"])
        with torch.no_grad():
            mem = transformer.encode(src, (src != 0).float())
        pose_t = torch.tensor(
            pose.reshape(1, -1, CFG["pose_dim"]), device=CFG["device"]
        )
        pose_refined = diffusion_obj.p_sample_loop(
            diffusion_refiner, pose_t, mem, start_t=30
        )
        pose = pose_refined[0].cpu().numpy().reshape(-1, 128, 3)

    # ── Post-process ──────────────────────────────────────────────────────────
    pose = normalise_pose(pose)
    pose = cosine_smooth(pose, window=CFG["smooth_window"])
    return pose


# =============================================================================
# SECTION 10 — SUBMISSION GENERATION
# =============================================================================
def generate_submission(
    test_df: pd.DataFrame,
    retriever: SemanticRetriever,
    transformer: ProgressiveTransformer,
    diffusion_refiner: Optional[DiffusionRefiner] = None,
    diffusion_obj: Optional[GaussianDiffusion] = None,
):
    """
    Generate pose sequences for all test sentences and save:
      - submission/  directory of {id}.npy files  (T, 128, 3)
      - submission.csv  with id + flattened metadata (if competition requires CSV)
    """
    out_dir = WORKING / "submission"
    out_dir.mkdir(exist_ok=True)

    records = []
    for i, row in test_df.iterrows():
        sid  = str(row["id"])
        text = str(row["text"])

        pose = hybrid_predict(
            text, retriever, transformer,
            diffusion_refiner, diffusion_obj,
        )

        # Save .npy
        np.save(str(out_dir / f"{sid}.npy"), pose)

        # Build flat CSV row (id + num_frames + mean_conf)
        records.append({
            "id"         : sid,
            "num_frames" : pose.shape[0],
            "mean_conf"  : float(pose[:, :, 2].mean()),
        })

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_df)}] id={sid}  frames={pose.shape[0]}")

    # CSV summary
    sub_df = pd.DataFrame(records)
    sub_df.to_csv(WORKING / "submission.csv", index=False)
    print(f"\n[Submission] saved {len(records)} poses → {out_dir}")
    print(sub_df.head())
    return sub_df


# =============================================================================
# SECTION 11 — MAIN ORCHESTRATION
# =============================================================================
def main():
    print("=" * 65)
    print("  Motion-S Text-to-Sign  |  Hybrid Pipeline")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_df, test_df = load_data()

    # Train / val split (90/10)
    val_frac = 0.10
    val_df   = train_df.sample(frac=val_frac, random_state=CFG["seed"])
    tr_df    = train_df.drop(val_df.index)
    print(f"[Split] train={len(tr_df)}  val={len(val_df)}")

    # ── 2. Tier 1: Retriever ──────────────────────────────────────────────────
    retriever = SemanticRetriever(alpha=0.4)
    retriever.fit(train_df)

    # Quick sanity check
    q = test_df["text"].iloc[0] if len(test_df) else "the weather is sunny"
    hits = retriever.retrieve(q, k=3)
    print(f"\n[Retriever] Query: '{q}'")
    for tid, s in hits:
        print(f"  → id={tid}  score={s:.3f}")

    # ── 3. Tier 2: Progressive Transformer ───────────────────────────────────
    print("\n[Transformer] Training …")
    transformer = train_transformer(tr_df, val_df)

    # Evaluate
    evaluate_on_val(transformer, val_df, n_samples=200)

    # ── 4. Tier 3: Diffusion (optional, skip on CPU) ─────────────────────────
    diffusion_refiner = None
    diffusion_obj     = None
    if CFG["device"] == "cuda":
        print("\n[Diffusion] Training refiner …")
        diffusion_obj     = GaussianDiffusion(CFG["diff_steps"],
                                              CFG["diff_beta_min"],
                                              CFG["diff_beta_max"])
        diffusion_refiner = train_diffusion(tr_df, transformer)
    else:
        print("\n[Diffusion] Skipped (no GPU). Transformer only.")

    # ── 5. Generate submission ────────────────────────────────────────────────
    print("\n[Inference] Generating test predictions …")
    generate_submission(
        test_df, retriever, transformer,
        diffusion_refiner, diffusion_obj,
    )

    print("\n✅  Pipeline complete.")
    print(f"   Outputs in {WORKING}")


if __name__ == "__main__":
    main()
