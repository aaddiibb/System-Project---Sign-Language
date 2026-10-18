# %% [markdown]
# # Motion-S / SignVRSE — Phase 7: Autoregressive Exposure-Bias Repair
#
# This experiment keeps Phase 3, the Phase-4 architecture, Phase-5 decoder, and Phase-6 length
# predictor fixed. It compares additional teacher forcing against two-pass self-conditioned point
# and span history corruption. This is **not** exact scheduled sampling.

# %%
from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from torch.utils.data import DataLoader, Dataset, Sampler

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_SPLITS = {"train": 9970, "validation": 1246, "test": 1247}
EXPECTED_SUPPORTED = {"train": 9936, "validation": 1243, "test": 1243}
LEVELS = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
RVQ_LEVELS, CODEBOOK_SIZE = 6, 512
PAD_ID, BOS_ID, EOS_ID, MOTION_VOCAB = 512, 513, 514, 515
MOTION_DIM, LATENT_DIM, TEMPORAL_EXPANSION = 668, 256, 4


@dataclass(frozen=True)
class Config:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    kaggle_working: str = "/kaggle/working"
    output_dir: str = "/kaggle/working/motion_s_phase7"
    max_t: int = 512
    learning_rate: float = 3e-5
    weight_decay: float = .01
    max_grad_norm: float = 1.0
    epochs: int = 10
    corruption_warmup_epochs: int = 5
    p_candidates: tuple[float, ...] = (.15, .30, .45)
    search_epochs: int = 3
    search_train_samples: int = 2048
    ar_subset_samples: int = 192
    ar_eval_interval: int = 2
    full_validation_candidates: int = 2
    span_min: int = 2
    span_max: int = 8
    warmup_ratio: float = .06
    near_zero_std: float = 1e-6


CFG = Config()
OUT = Path(CFG.output_dir)
DIAG, MODELS, TRAINING, TOKENS, DECODED, RESULTS, FIGURES = (
    OUT / "diagnostics", OUT / "models", OUT / "training", OUT / "tokens", OUT / "decoded", OUT / "results", OUT / "figures")
for path in (OUT, DIAG, MODELS, TRAINING, TOKENS / "oracle_length", TOKENS / "predicted_length",
             DECODED / "oracle_length", DECODED / "predicted_length", RESULTS, FIGURES): path.mkdir(parents=True, exist_ok=True)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


seed_all(CFG.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda": raise RuntimeError("Phase 7 requires a Kaggle GPU")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle: json.dump(safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: return json.load(handle)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def canon(value: str) -> str: return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def spearman(x, y) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 2 or frame.x.nunique() < 2 or frame.y.nunique() < 2: return float("nan")
    return float(frame.x.rank().corr(frame.y.rank()))


print("Python", sys.version.split()[0], "PyTorch", torch.__version__, "GPU", torch.cuda.get_device_name(0))
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 7.0 — Artifact discovery and immutable dependency verification

# %%
SEARCH = [Path(CFG.kaggle_input), Path(CFG.kaggle_working)]


def roots(predicate) -> list[Path]:
    found = []
    for base in SEARCH:
        if not base.exists(): continue
        for current, directories, _ in os.walk(base):
            candidate = Path(current)
            if candidate.resolve() == OUT.resolve(): directories[:] = []; continue
            if predicate(candidate): found.append(candidate.resolve()); directories[:] = []
    return sorted(dict.fromkeys(found), key=str)


def choose(label: str, predicate, identities: list[str]) -> Path:
    candidates = roots(predicate)
    if not candidates: raise RuntimeError(f"Missing {label}")
    signatures = {root: tuple((rel, sha(root / rel)) for rel in identities if (root / rel).is_file()) for root in candidates}
    if len(set(signatures.values())) != 1: raise RuntimeError(f"Ambiguous non-identical {label}: {[(str(k), v) for k,v in signatures.items()]}")
    preferred = [p for p in candidates if str(p).startswith(CFG.kaggle_working)]
    selected = sorted(preferred or candidates, key=lambda p: (len(p.parts), str(p)))[0]
    print(f"{label}: {selected}"); return selected


P1 = choose("Phase 1", lambda p: (p/"dataset_config.json").is_file() and (p/"splits/train_split.csv").is_file(),
            ["dataset_config.json", "file_manifest.csv"])
P3 = choose("Phase 3", lambda p: all((p/x).is_file() for x in ("phase3_report.json", "best_model/config.json",
            "tokenizer/tokenizer_config.json", "results/validation_predictions.csv", "results/test_predictions.csv")),
            ["phase3_report.json", "best_model/config.json", "best_model/model.safetensors", "tokenizer/tokenizer_config.json"])
P4 = choose("Phase 4C", lambda p: all((p/x).is_file() for x in ("phase4c_report.json", "best_model/model.pt",
            "best_model/model_config.json", "best_model/token_config.json")),
            ["phase4c_report.json", "best_model/model.pt", "best_model/model_config.json"])
P5 = choose("Phase 5B", lambda p: all((p/x).is_file() for x in ("phase5b_report.json", "diagnostics/decoder_contract.json",
            "diagnostics/motion_mean_668.npy", "diagnostics/motion_std_668.npy", "diagnostics/empirical_motion_thresholds.json")),
            ["phase5b_report.json", "diagnostics/decoder_contract.json", "diagnostics/motion_mean_668.npy"])
P6 = choose("Phase 6", lambda p: all((p/x).is_file() for x in ("phase6_report.json", "length_model/best_model.pt",
            "length_model/config.json", "tokens/oracle_length_test.npz", "tokens/predicted_length_end2end_test.npz",
            "tokens/oracle_length_validation.npz", "tokens/predicted_length_end2end_validation.npz")),
            ["phase6_report.json", "length_model/best_model.pt", "length_model/config.json"])

P1CFG, P4REPORT, P6REPORT = read_json(P1/"dataset_config.json"), read_json(P4/"phase4c_report.json"), read_json(P6/"phase6_report.json")
if P1CFG.get("split_fingerprint_sha256") != FINGERPRINT: raise RuntimeError("Phase-1 fingerprint mismatch")
if P4REPORT.get("phase4c_complete") is not True or P6REPORT.get("phase6_complete") is not True: raise RuntimeError("Phase 4C/6 incomplete")
ID, GLOSS, SENTENCE = P1CFG["id_column"], P1CFG["gloss_column"], P1CFG.get("sentence_column", "sentence")
TOKEN_COLS = list(P1CFG["motion_token_columns"])
if TOKEN_COLS != LEVELS: raise RuntimeError(f"RVQ level-order mismatch: {TOKEN_COLS}")


def unique_file(label: str, name: str, expected_hash: Optional[str] = None) -> Path:
    candidates = sorted({p.resolve() for base in SEARCH if base.exists() for p in base.rglob(name) if p.is_file()}, key=str)
    if expected_hash: candidates = [p for p in candidates if sha(p) == expected_hash]
    if not candidates: raise RuntimeError(f"Missing {label}: {name}")
    hashes = {p: sha(p) for p in candidates}
    if len(set(hashes.values())) != 1: raise RuntimeError(f"Ambiguous {label}: {[(str(p), h) for p,h in hashes.items()]}")
    print(f"{label}: {candidates[0]}"); return candidates[0]


decoder_contract = read_json(P5/"diagnostics/decoder_contract.json")
RVQ_CKPT = unique_file("official RVQ checkpoint", "rvq_vae_best.pth", decoder_contract.get("checkpoint_sha256"))


def locate_train() -> Path:
    candidates = []
    for base in SEARCH:
        if not base.exists(): continue
        for path in base.rglob("train.csv"):
            try:
                if {ID, GLOSS, *TOKEN_COLS}.issubset(pd.read_csv(path, nrows=0).columns): candidates.append(path.resolve())
            except Exception: pass
    if not candidates: raise RuntimeError("Official train.csv missing")
    hashes = {p: sha(p) for p in candidates}
    if len(set(hashes.values())) != 1: raise RuntimeError(f"Ambiguous train.csv: {[(str(p),h) for p,h in hashes.items()]}")
    print("Official train.csv:", candidates[0]); return candidates[0]


def locate_features() -> Path:
    candidates = sorted({(Path(c)/d).resolve() for base in SEARCH if base.exists() for c,ds,_ in os.walk(base)
                         for d in ds if canon(d)=="motionfeatures"}, key=str)
    if len(candidates) != 1: raise RuntimeError(f"Expected one Motion-Features directory: {[str(p) for p in candidates]}")
    print("Motion-Features:", candidates[0]); return candidates[0]


TRAIN_CSV, FEATURE_DIR = locate_train(), locate_features()
P3_MODEL, P3_TOKENIZER, P4_CKPT = P3/"best_model", P3/"tokenizer", P4/"best_model/model.pt"
P6_LENGTH_CKPT, P6_LENGTH_CONFIG = P6/"length_model/best_model.pt", P6/"length_model/config.json"
resolved = {"phase1":P1,"phase3":P3,"phase4":P4,"phase5":P5,"phase6":P6,"train_csv":TRAIN_CSV,
            "motion_features":FEATURE_DIR,"phase4_checkpoint":P4_CKPT,"phase4_sha256":sha(P4_CKPT),
            "length_checkpoint":P6_LENGTH_CKPT,"length_sha256":sha(P6_LENGTH_CKPT),
            "rvq_checkpoint":RVQ_CKPT,"rvq_sha256":sha(RVQ_CKPT)}
for key,value in resolved.items(): print(key, ":", value)
write_json(DIAG/"artifact_verification.json", {**resolved,"phase1_fingerprint":FINGERPRINT,"passed":True})

# %% [markdown]
# ## 7.1 — Locked data, models, decoder, and Phase-6 baseline reproduction

# %%
def parse_cell(value: Any) -> np.ndarray:
    text = str(value).strip()
    if not text or text.casefold() in {"nan","none","null"}: raise ValueError("blank token cell")
    try: parsed = json.loads(text)
    except (json.JSONDecodeError,TypeError):
        try: parsed = ast.literal_eval(text)
        except (ValueError,SyntaxError): parsed = [int(x) for x in re.split(r"[\s,]+",text) if x]
    array = np.asarray(parsed)
    if array.dtype==object or array.ndim!=1 or array.size==0: raise ValueError(f"bad token shape {array.shape}")
    numeric=array.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric,np.floor(numeric)).all(): raise ValueError("non-integer token")
    return numeric.astype(np.int64)


official = pd.read_csv(TRAIN_CSV,dtype={ID:str},low_memory=False); official[ID]=official[ID].astype(str)
official_index = official.set_index(ID,drop=False)
TOKEN_CACHE:dict[str,np.ndarray]={}
def tokens_for(sample_id: str) -> np.ndarray:
    sample_id=str(sample_id)
    if sample_id in TOKEN_CACHE:return TOKEN_CACHE[sample_id]
    result=np.stack([parse_cell(official_index.loc[sample_id,c]) for c in TOKEN_COLS])
    if result.min()<0 or result.max()>511: raise ValueError(sample_id)
    TOKEN_CACHE[sample_id]=result
    return result


def split_file(name: str) -> Path:
    names={"train":["train_split.csv"],"validation":["val_split.csv","validation_split.csv"],"test":["test_split.csv"]}[name]
    found=[P1/"splits"/x for x in names if (P1/"splits"/x).is_file()]
    if len(found)!=1: raise RuntimeError(f"{name} split ambiguity {found}")
    return found[0]


frames={}
for name,expected in EXPECTED_SPLITS.items():
    frame=pd.read_csv(split_file(name),dtype={ID:str},low_memory=False); frame[ID]=frame[ID].astype(str)
    if len(frame)!=expected or not frame[ID].is_unique: raise RuntimeError(f"{name} locked split mismatch")
    lengths=np.asarray([tokens_for(x).shape[1] for x in frame[ID]],dtype=int)
    frame["motion_length"]=lengths; frame=frame.loc[frame.motion_length<=CFG.max_t].reset_index(drop=True)
    if len(frame)!=EXPECTED_SUPPORTED[name]: raise RuntimeError(f"{name} supported count mismatch")
    frames[name]=frame
train_frame,val_frame,test_frame=frames["train"],frames["validation"],frames["test"]

try:
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
except ImportError as error: raise ImportError("Transformers unavailable") from error
tokenizer=AutoTokenizer.from_pretrained(P3_TOKENIZER,local_files_only=True,use_fast=True)
seq2seq=AutoModelForSeq2SeqLM.from_pretrained(P3_MODEL,local_files_only=True)
encoder=seq2seq.get_encoder().to(DEVICE).eval(); del seq2seq
for parameter in encoder.parameters(): parameter.requires_grad=False
MODEL_CONFIG=read_json(P4/"best_model/model_config.json")


class MotionModel(nn.Module):
    def __init__(self,gloss_encoder):
        super().__init__(); self.gloss_encoder=gloss_encoder
        dm,dl=int(MODEL_CONFIG["d_model"]),int(MODEL_CONFIG["d_level"])
        self.encoder_projection=nn.Linear(int(gloss_encoder.config.d_model),dm)
        self.level_embeddings=nn.ModuleList([nn.Embedding(515,dl,padding_idx=PAD_ID) for _ in range(6)])
        self.motion_projection=nn.Linear(6*dl,dm); self.temporal_positions=nn.Embedding(int(MODEL_CONFIG["max_model_positions"]),dm)
        layer=nn.TransformerDecoderLayer(dm,int(MODEL_CONFIG["num_heads"]),int(MODEL_CONFIG["ffn_dimension"]),
            float(MODEL_CONFIG["dropout"]),activation="gelu",batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,int(MODEL_CONFIG["decoder_layers"]),norm=nn.LayerNorm(dm))
        self.heads=nn.ModuleList([nn.Linear(dm,515) for _ in range(6)])
    def train(self,mode=True): super().train(mode); self.gloss_encoder.eval(); return self
    def encode(self,ids,mask):
        with torch.no_grad(): hidden=self.gloss_encoder(input_ids=ids,attention_mask=mask).last_hidden_state
        return self.encoder_projection(hidden)
    def decode(self,tokens,memory,gloss_mask,padding):
        emb=torch.cat([self.level_embeddings[l](tokens[:,l]) for l in range(6)],-1)
        emb=self.motion_projection(emb)+self.temporal_positions(torch.arange(tokens.shape[-1],device=tokens.device)[None])
        causal=torch.triu(torch.ones(tokens.shape[-1],tokens.shape[-1],dtype=torch.bool,device=tokens.device),1)
        hidden=self.decoder(emb,memory,tgt_mask=causal,tgt_key_padding_mask=padding,memory_key_padding_mask=~gloss_mask.bool())
        return torch.stack([head(hidden) for head in self.heads],1)
    def forward(self,ids,mask,tokens,padding): return self.decode(tokens,self.encode(ids,mask),mask,padding)


def fresh_model(checkpoint_path: Path=P4_CKPT) -> MotionModel:
    model=MotionModel(encoder).to(DEVICE); payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    state=payload["motion_state_dict"] if "motion_state_dict" in payload else payload
    incompatible=model.load_state_dict(state,strict=False)
    expected=[n for n in model.state_dict() if n.startswith("gloss_encoder.")]
    if sorted(incompatible.missing_keys)!=sorted(expected) or incompatible.unexpected_keys: raise RuntimeError(str(incompatible))
    for name,p in model.named_parameters(): p.requires_grad=not name.startswith("gloss_encoder.")
    return model


class MotionDataset(Dataset):
    def __init__(self,frame:pd.DataFrame,gloss_col:str=GLOSS): self.frame=frame.reset_index(drop=True); self.gloss_col=gloss_col;self.cache={}
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        if i not in self.cache:self.cache[i]=tokens_for(str(row[ID]))
        return {"sample_id":str(row[ID]),"gloss":str(row[self.gloss_col]),
            "gt_gloss":str(row[GLOSS]),"tokens":self.cache[i],"length":int(row.motion_length)}


def collate(items):
    encoded=tokenizer([x["gloss"] for x in items],padding=True,truncation=False,return_tensors="pt")
    maximum=max(x["length"] for x in items)+1; batch=len(items)
    decoder=torch.full((batch,6,maximum),PAD_ID,dtype=torch.long); targets=torch.full_like(decoder,PAD_ID)
    padding=torch.ones((batch,maximum),dtype=torch.bool)
    for i,item in enumerate(items):
        length=item["length"]; target=torch.from_numpy(item["tokens"])
        decoder[i,:,0]=BOS_ID; decoder[i,:,1:length+1]=target; targets[i,:,:length]=target; targets[i,:,length]=EOS_ID; padding[i,:length+1]=False
    return {"sample_id":[x["sample_id"] for x in items],"gloss":[x["gloss"] for x in items],
        "ids":encoded["input_ids"],"mask":encoded["attention_mask"],"decoder":decoder,"targets":targets,
        "padding":padding,"lengths":torch.tensor([x["length"] for x in items])}


BOUNDARIES=(64,96,128,181,256,384,512); BATCH_SIZES=(8,8,8,8,4,2,1)
class BucketSampler(Sampler[list[int]]):
    def __init__(self,lengths,shuffle,seed):
        self.buckets=[[] for _ in BOUNDARIES]; self.shuffle=shuffle; self.seed=seed; self.epoch=0
        for i,length in enumerate(lengths): self.buckets[next(j for j,u in enumerate(BOUNDARIES) if length<=u)].append(i)
    def set_epoch(self,e): self.epoch=e
    def batches(self):
        rng=np.random.default_rng(self.seed+self.epoch); result=[]
        for bucket,size in zip(self.buckets,BATCH_SIZES):
            values=list(bucket)
            if self.shuffle:rng.shuffle(values)
            result.extend(values[i:i+size] for i in range(0,len(values),size))
        if self.shuffle:rng.shuffle(result)
        return result
    def __iter__(self): return iter(self.batches())
    def __len__(self): return sum(math.ceil(len(b)/s) for b,s in zip(self.buckets,BATCH_SIZES))


def loader(frame,gloss_col=GLOSS,shuffle=False,seed=42):
    sampler=BucketSampler(frame.motion_length.astype(int).tolist(),shuffle,seed)
    return DataLoader(MotionDataset(frame,gloss_col),batch_sampler=sampler,collate_fn=collate,num_workers=0,pin_memory=True),sampler


def move(batch): return {k:(v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


class Packed:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z: self.ids=z["sample_ids"].astype(str);self.lengths=z["lengths"].astype(int);self.offsets=z["offsets"].astype(int);self.flat=z["tokens_flat"].astype(np.int64)
        if self.flat.shape[0]!=6 or not np.array_equal(np.diff(self.offsets),self.lengths): raise RuntimeError(path)
        self.index={x:i for i,x in enumerate(self.ids)}
    def get(self,sid): i=self.index[str(sid)];return self.flat[:,self.offsets[i]:self.offsets[i+1]]


P6_ORACLE_VAL=Packed(P6/"tokens/oracle_length_validation.npz");P6_END_VAL=Packed(P6/"tokens/predicted_length_end2end_validation.npz")
reference_values={"aligned":.144048,"feature":.856556,"collapse":.510861}
measured={"aligned":float(P6REPORT["token_space"]["oracle_gt_length"]["aligned_token_accuracy"]),
          "feature":float(P6REPORT["oracle_length"]["normalized_feature_mae"]),
          "collapse":float(P6REPORT["oracle_length"]["collapse_rate"])}
report_compatible=(abs(measured["aligned"]-reference_values["aligned"])<=.03 and
                   abs(measured["feature"]-reference_values["feature"])<=.10 and
                   abs(measured["collapse"]-reference_values["collapse"])<=.10)
baseline_reproduction={"phase6_report_values":measured,"reference_values":reference_values,"report_compatible":report_compatible,
    "phase4_checkpoint_sha256":sha(P4_CKPT),"phase6_recorded_phase4_sha256":read_json(P6/"diagnostics/artifact_verification.json").get("phase4_checkpoint_sha256")}
baseline_reproduction["checkpoint_identity"]=(baseline_reproduction["phase6_recorded_phase4_sha256"] in (None,sha(P4_CKPT)))
baseline_reproduction["passed"]=bool(report_compatible and baseline_reproduction["checkpoint_identity"])
write_json(DIAG/"baseline_reproduction.json",baseline_reproduction)
if not baseline_reproduction["passed"]: raise RuntimeError("Phase-6 baseline reproduction failed; inspect diagnostics/baseline_reproduction.json")

# %% [markdown]
# ## 7.2–7.10 — Two-pass corruption, matched optimization, and validation-only policy

# %%
def loss_info(logits,targets):
    losses=[];correct=np.zeros(6);total=np.zeros(6)
    predictions=logits.argmax(-1)
    for level in range(6):
        losses.append(F.cross_entropy(logits[:,level].reshape(-1,515),targets[:,level].reshape(-1),ignore_index=PAD_ID))
        valid=targets[:,level]<512;correct[level]=int((predictions[:,level][valid]==targets[:,level][valid]).sum());total[level]=int(valid.sum())
    return torch.stack(losses).mean(),correct,total


def point_mask(valid:torch.Tensor,p:float,generator:torch.Generator)->torch.Tensor:
    return (torch.rand(valid.shape,generator=generator,device="cpu").to(valid.device)<p)&valid


def span_mask(valid:torch.Tensor,p:float,generator:torch.Generator)->torch.Tensor:
    output=torch.zeros_like(valid)
    for i in range(len(valid)):
        valid_count=int(valid[i].sum()); target=round(p*valid_count); covered=0;attempts=0
        while covered<target and valid_count and attempts<max(20,4*target):
            attempts+=1
            span=int(torch.randint(CFG.span_min,CFG.span_max+1,(1,),generator=generator))
            start=int(torch.randint(0,max(valid.shape[1]-span+1,1),(1,),generator=generator))
            before=int(output[i].sum());output[i,start:start+span]|=valid[i,start:start+span];covered+=int(output[i].sum())-before
    return output


def corrupted_history(model,batch,mode,p,seed):
    if mode=="control" or p<=0:return batch["decoder"]
    model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=AMP_DTYPE): logits=model(batch["ids"],batch["mask"],batch["decoder"],batch["padding"])
    predicted=logits[:,:,:-1,:512].argmax(-1);del logits
    valid=~batch["padding"][:,1:];generator=torch.Generator().manual_seed(seed)
    selected=point_mask(valid,p,generator) if mode=="point" else span_mask(valid,p,generator)
    corrupted=batch["decoder"].clone(); replacement=predicted
    corrupted[:,:,1:]=torch.where(selected[:,None,:],replacement,corrupted[:,:,1:])
    return corrupted


@torch.no_grad()
def teacher_evaluate(model,frame):
    data,_=loader(frame,shuffle=False,seed=99);model.eval();loss_num=0.;count=0;correct=np.zeros(6);total=np.zeros(6)
    for raw in data:
        batch=move(raw)
        with torch.autocast("cuda",dtype=AMP_DTYPE): logits=model(batch["ids"],batch["mask"],batch["decoder"],batch["padding"]);loss,c,t=loss_info(logits,batch["targets"])
        n=int((batch["targets"]!=PAD_ID).sum());loss_num+=float(loss)*n;count+=n;correct+=c;total+=t
    return {"loss":loss_num/max(count,1),"motion_only_accuracy":float(correct.sum()/np.maximum(total.sum(),1)),"per_level_accuracy":(correct/np.maximum(total,1)).tolist()}


@torch.inference_mode()
def exact_generate(model,frame,gloss_col=GLOSS,lengths:Optional[np.ndarray]=None,label="generate"):
    lengths=frame.motion_length.to_numpy(int) if lengths is None else np.asarray(lengths,int);order=np.argsort(lengths,kind="stable");outputs={}
    cursor=0;model.eval()
    while cursor<len(order):
        first=int(lengths[order[cursor]]);size=8 if first<=181 else 4 if first<=256 else 2 if first<=384 else 1
        indices=order[cursor:cursor+size];cursor+=len(indices);texts=frame.iloc[indices][gloss_col].astype(str).tolist();lens=lengths[indices]
        encoded=tokenizer(texts,padding=True,truncation=False,return_tensors="pt");ids=encoded["input_ids"].to(DEVICE);mask=encoded["attention_mask"].to(DEVICE)
        with torch.autocast("cuda",dtype=AMP_DTYPE):memory=model.encode(ids,mask)
        history=torch.full((len(indices),6,1),BOS_ID,dtype=torch.long,device=DEVICE);padding=torch.zeros((len(indices),1),dtype=torch.bool,device=DEVICE);steps=[]
        for _ in range(int(lens.max())):
            with torch.autocast("cuda",dtype=AMP_DTYPE): choice=model.decode(history,memory,mask,padding)[:,:,-1,:512].argmax(-1)
            steps.append(choice.cpu());history=torch.cat([history,choice.unsqueeze(-1)],-1);padding=torch.cat([padding,torch.zeros((len(indices),1),dtype=torch.bool,device=DEVICE)],1)
        stacked=torch.stack(steps,-1).numpy().astype(np.int64)
        for j,index in enumerate(indices):outputs[str(frame.iloc[index][ID])]=stacked[j,:,:int(lens[j])]
        if len(outputs)%100<len(indices):print(label,len(outputs),"/",len(frame))
    return outputs


try:
    from rapidfuzz.distance import Levenshtein
    def edit_distance(a,b):return int(Levenshtein.distance(a,b))
except ImportError:
    def edit_distance(a,b):
        previous=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            current=[i]
            for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+int(x!=y)))
            previous=current
        return previous[-1]


def token_metrics(frame,mapping,label):
    rows=[];lc=np.zeros(6);lt=np.zeros(6);micro_c=micro_t=0
    for sid in frame[ID].astype(str):
        gt=tokens_for(sid);pred=mapping[sid];aligned=min(gt.shape[1],pred.shape[1]);per=[];ed=[]
        for level in range(6):
            c=int((gt[level,:aligned]==pred[level,:aligned]).sum());lc[level]+=c;lt[level]+=aligned;per.append(c/max(aligned,1));ed.append(edit_distance(gt[level].tolist(),pred[level].tolist())/max(gt.shape[1],pred.shape[1],1))
        correct=int((gt[:,:aligned]==pred[:,:aligned]).sum());micro_c+=correct;micro_t+=6*aligned
        rows.append({"sample_id":sid,"model":label,"aligned_token_accuracy":correct/max(6*aligned,1),"normalized_token_edit_distance":float(np.mean(ed)),**{f"level_{i}_accuracy":per[i] for i in range(6)}})
    detail=pd.DataFrame(rows);summary={"model":label,"aligned_token_accuracy":float(detail.aligned_token_accuracy.mean()),
        "micro_token_accuracy":micro_c/max(micro_t,1),"normalized_token_edit_distance":float(detail.normalized_token_edit_distance.mean()),
        "per_level_accuracy":(lc/np.maximum(lt,1)).tolist()}
    return summary,detail


rng=np.random.default_rng(CFG.seed);quantiles=pd.qcut(val_frame.motion_length,10,labels=False,duplicates="drop")
subset_indices=[]
for _,indices in val_frame.groupby(quantiles).groups.items(): subset_indices.extend(rng.choice(list(indices),size=min(math.ceil(CFG.ar_subset_samples/10),len(indices)),replace=False))
VAL_SUBSET=val_frame.loc[sorted(set(subset_indices))].iloc[:CFG.ar_subset_samples].reset_index(drop=True)
SEARCH_TRAIN=train_frame.iloc[np.linspace(0,len(train_frame)-1,CFG.search_train_samples).round().astype(int)].reset_index(drop=True)
selection_policy={"primary":"lexicographic","order":["lower collapse_rate","lower normalized_token_edit_distance","higher micro_token_accuracy","lower normalized_feature_mae"],
    "epoch_screen":"fixed deterministic oracle-length validation subset; lower edit then higher micro accuracy",
    "full_validation":"top two screened epochs per definitive run","test_used":False}
write_json(DIAG/"model_selection_policy.json",selection_policy)

# Actual reloaded-model reproduction uses validation only, preserving the untouched-test rule.
# Full Phase-6 test reference values above are read from its already-frozen report, never tuned on.
baseline_indices=np.linspace(0,len(val_frame)-1,24).round().astype(int)
baseline_subset=val_frame.sort_values(["motion_length",ID]).iloc[baseline_indices].reset_index(drop=True)
baseline_model=fresh_model();baseline_generated=exact_generate(baseline_model,baseline_subset,label="Phase-6 validation reproduction")
baseline_saved_oracle={sid:P6_ORACLE_VAL.get(sid) for sid in baseline_subset[ID].astype(str)}
baseline_saved_end={sid:P6_END_VAL.get(sid) for sid in baseline_subset[ID].astype(str)}
regen_summary,_=token_metrics(baseline_subset,baseline_generated,"reloaded_phase4")
saved_summary,_=token_metrics(baseline_subset,baseline_saved_oracle,"saved_phase6_oracle")
saved_end_summary,_=token_metrics(baseline_subset,baseline_saved_end,"saved_phase6_end2end")
subset_compatible=(abs(regen_summary["aligned_token_accuracy"]-saved_summary["aligned_token_accuracy"])<=.05 and
                   abs(regen_summary["normalized_token_edit_distance"]-saved_summary["normalized_token_edit_distance"])<=.08 and
                   all(baseline_generated[sid].shape[1]==tokens_for(sid).shape[1] for sid in baseline_subset[ID].astype(str)))
baseline_reproduction.update({"reproduction_split":"validation","deterministic_subset_ids":baseline_subset[ID].astype(str).tolist(),
    "reloaded_oracle_length_subset":regen_summary,"saved_phase6_oracle_subset":saved_summary,
    "saved_phase6_predicted_length_end2end_subset":saved_end_summary,"subset_compatible":subset_compatible,
    "all_regenerated_ids_valid":all(x.min()>=0 and x.max()<=511 for x in baseline_generated.values())})
baseline_reproduction["passed"]=bool(baseline_reproduction["passed"] and subset_compatible and baseline_reproduction["all_regenerated_ids_valid"])
write_json(DIAG/"baseline_reproduction.json",baseline_reproduction)
del baseline_model,baseline_generated;gc.collect();torch.cuda.empty_cache()
if not baseline_reproduction["passed"]:raise RuntimeError("Reloaded Phase-6 validation baseline reproduction failed")


def schedule_probability(epoch,pmax,warmup): return pmax*min(1.,max(0.,(epoch-1)/max(warmup-1,1)))
write_json(DIAG/"corruption_schedule.json",{"name":"two-pass self-conditioned history corruption","not_exact_scheduled_sampling":True,
    "candidates":CFG.p_candidates,"formula":"p_max * min(1,(epoch-1)/(warmup_epochs-1))","warmup_epochs":CFG.corruption_warmup_epochs,
    "point":"temporal Bernoulli; all six levels together","span":[CFG.span_min,CFG.span_max]})


def train_epochs(mode,pmax,frame,epochs,screen=True,save_candidates=False):
    seed_all(CFG.seed);model=fresh_model();data,sampler=loader(frame,shuffle=True,seed=CFG.seed)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CFG.learning_rate,weight_decay=CFG.weight_decay)
    total=max(1,len(data)*epochs);warm=max(1,round(total*CFG.warmup_ratio))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warm,max(0.,(total-s)/max(total-warm,1))))
    try:scaler=torch.amp.GradScaler("cuda",enabled=AMP_DTYPE==torch.float16)
    except TypeError:scaler=torch.cuda.amp.GradScaler(enabled=AMP_DTYPE==torch.float16)
    history=[];candidates=[];step=0
    for epoch in range(1,epochs+1):
        sampler.set_epoch(epoch);model.train();losses=[];p=schedule_probability(epoch,pmax,min(CFG.corruption_warmup_epochs,epochs))
        for batch_number,raw in enumerate(data):
            batch=move(raw);corrupted=corrupted_history(model,batch,mode,p,CFG.seed+epoch*100000+batch_number);model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=AMP_DTYPE):logits=model(batch["ids"],batch["mask"],corrupted,batch["padding"]);loss,_,_=loss_info(logits,batch["targets"])
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],CFG.max_grad_norm)
            scaler.step(optimizer);scaler.update();scheduler.step();step+=1;losses.append(float(loss.detach()))
        tf=teacher_evaluate(model,val_frame);row={"epoch":epoch,"p_corruption":p,"train_loss":float(np.mean(losses)),
            "validation_loss":tf["loss"],"validation_motion_accuracy":tf["motion_only_accuracy"],"optimizer_steps":step}
        if screen and (epoch%CFG.ar_eval_interval==0 or epoch==epochs):
            mapping=exact_generate(model,VAL_SUBSET,label=f"{mode} epoch {epoch} subset");score,_=token_metrics(VAL_SUBSET,mapping,mode)
            row.update({"subset_edit":score["normalized_token_edit_distance"],"subset_micro":score["micro_token_accuracy"]})
            state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("gloss_encoder.")}
            candidates.append({"epoch":epoch,"edit":score["normalized_token_edit_distance"],"micro":score["micro_token_accuracy"],"state":state})
        history.append(row);print(mode,"epoch",epoch,row)
    if not candidates:
        state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("gloss_encoder.")};candidates=[{"epoch":epochs,"edit":99.,"micro":0.,"state":state}]
    candidates.sort(key=lambda x:(x["edit"],-x["micro"]));return model,pd.DataFrame(history),candidates


search_rows=[]
for mode in ("point","span"):
    for pmax in CFG.p_candidates:
        model,history,candidates=train_epochs(mode,pmax,SEARCH_TRAIN,CFG.search_epochs,screen=True)
        best=candidates[0];search_rows.append({"mode":mode,"p_max":pmax,"validation_subset_edit":best["edit"],"validation_subset_micro":best["micro"]})
        del model,candidates;gc.collect();torch.cuda.empty_cache()
search_frame=pd.DataFrame(search_rows);search_frame.to_csv(TRAINING/"corruption_probability_search.csv",index=False)
selected_p={mode:float(search_frame.query("mode==@mode").sort_values(["validation_subset_edit","validation_subset_micro"],ascending=[True,False]).iloc[0].p_max) for mode in ("point","span")}
print("Validation-selected corruption maxima:",selected_p)

# %% [markdown]
# ## 7.11–7.16 — Matched definitive training and full-validation selection

# %%
motion_mean=np.load(P5/"diagnostics/motion_mean_668.npy",allow_pickle=False).astype(np.float32)
motion_std=np.load(P5/"diagnostics/motion_std_668.npy",allow_pickle=False).astype(np.float32);active=motion_std>=CFG.near_zero_std;safe_std=motion_std.copy();safe_std[~active]=1
thresholds=read_json(P5/"diagnostics/empirical_motion_thresholds.json")


class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.decoder=nn.Sequential(nn.ConvTranspose1d(256,512,3,2,1,output_padding=1),nn.ReLU(),nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512,512,3,2,1,output_padding=1),nn.ReLU(),nn.BatchNorm1d(512),nn.ConvTranspose1d(512,512,3,1,1),nn.ReLU(),nn.BatchNorm1d(512),nn.ConvTranspose1d(512,668,3,1,1))
    def forward(self,x):return self.decoder(x)


rvq_payload=torch.load(RVQ_CKPT,map_location="cpu",weights_only=True)
for key in ("model_state_dict","state_dict"):
    if isinstance(rvq_payload,dict) and key in rvq_payload:rvq_payload=rvq_payload[key];break
rvq_state={k:v for k,v in rvq_payload.items() if isinstance(v,torch.Tensor)};decoder=Decoder()
decoder.load_state_dict({k[len("decoder."):]:v for k,v in rvq_state.items() if k.startswith("decoder.")},strict=True);decoder=decoder.to(DEVICE).eval()
codebooks=torch.stack([rvq_state[f"rvq.quantizers.{i}.embedding"].float() for i in range(6)]).to(DEVICE)
mean_t=torch.from_numpy(motion_mean).to(DEVICE);std_t=torch.from_numpy(safe_std).to(DEVICE)
feature_index={p.stem:p for p in FEATURE_DIR.glob("*.npy")}


@torch.inference_mode()
def decode_one(tokens):
    ids=torch.as_tensor(tokens,dtype=torch.long,device=DEVICE);latent=torch.zeros((256,tokens.shape[1]),device=DEVICE)
    for level in range(6):latent+=codebooks[level][:,ids[level]]
    motion=(decoder(latent[None]).squeeze(0).T*std_t+mean_t).cpu().numpy().astype(np.float32)
    if motion.shape!=(4*tokens.shape[1],668) or not np.isfinite(motion).all():raise RuntimeError("decode failure")
    return motion


def z(m):return ((m[:,active]-motion_mean[active])/motion_std[active]).astype(np.float32)
def resample(m,n):return F.interpolate(torch.from_numpy(m.T)[None],size=n,mode="linear",align_corners=True).squeeze(0).T.numpy() if len(m)!=n else m
def velocity(m):
    value=z(m);return np.sqrt(np.mean(np.square(np.diff(value,axis=0),dtype=np.float64),axis=1)) if len(value)>1 else np.empty(0)
def repetition(values,max_lag=180):
    x=np.asarray(values,dtype=np.float64);x=x-x.mean() if len(x) else x
    if len(x)<6:return float("nan")
    values=[]
    for lag in range(2,min(max_lag,len(x)//2)+1):
        a,b=x[:-lag],x[lag:];d=np.sqrt(np.dot(a,a)*np.dot(b,b))
        if d>0:values.append(float(np.dot(a,b)/d))
    return max(values) if values else float("nan")


def motion_metrics(frame,mapping,label,save_dir:Optional[Path]=None):
    rows=[]
    if save_dir:save_dir.mkdir(parents=True,exist_ok=True)
    for number,sid in enumerate(frame[ID].astype(str),1):
        pred=decode_one(mapping[sid]);gt=np.load(feature_index[sid],allow_pickle=False).astype(np.float32);aligned=resample(pred,len(gt));zp,zg=z(aligned),z(gt)
        vel=velocity(pred);mean_vel=float(vel.mean()) if len(vel) else 0.;variance=float(np.mean(np.var(z(pred),axis=0,dtype=np.float64)))
        maxz=float(np.max(np.abs(z(pred))));collapse=mean_vel<float(thresholds["static_velocity_threshold_gt_percentile_1"]) or variance<float(thresholds["static_variance_threshold_gt_percentile_1"])
        extreme=float(pred.min())<float(thresholds["extreme_minimum_threshold_gt_sample_min_percentile_0_1"]) or float(pred.max())>float(thresholds["extreme_maximum_threshold_gt_sample_max_percentile_99_9"]) or maxz>float(thresholds["extreme_normalized_magnitude_threshold_gt_percentile_99_9"])
        rows.append({"sample_id":sid,"model":label,"normalized_feature_mae":float(np.mean(np.abs(zp-zg))),
            "time_normalized_raw_mae":float(np.mean(np.abs(aligned-gt))),"velocity_mae":float(np.mean(np.abs(np.diff(zp,axis=0)-np.diff(zg,axis=0)))),
            "motion_variance":variance,"repetition_score":repetition(vel),"collapse":collapse,"extreme":extreme,
            "gt_frames":len(gt),"pred_frames":len(pred),"duration_error":len(pred)-len(gt)})
        if save_dir:np.save(save_dir/f"{sid}.npy",pred,allow_pickle=False)
        if number%100==0:print(label,"motion",number,"/",len(frame))
    detail=pd.DataFrame(rows);summary={"model":label,"normalized_feature_mae":float(detail.normalized_feature_mae.mean()),
        "time_normalized_raw_mae":float(detail.time_normalized_raw_mae.mean()),"velocity_mae":float(detail.velocity_mae.mean()),
        "motion_variance":float(detail.motion_variance.mean()),"repetition_score":float(detail.repetition_score.mean()),
        "collapse_rate":float(detail.collapse.mean()),"extreme_motion_rate":float(detail.extreme.mean()),
        "duration_mae":float(detail.duration_error.abs().mean())}
    return summary,detail


def load_state_into(model,state):
    incompatible=model.load_state_dict(state,strict=False);expected=[n for n in model.state_dict() if n.startswith("gloss_encoder.")]
    if sorted(incompatible.missing_keys)!=sorted(expected) or incompatible.unexpected_keys:raise RuntimeError(str(incompatible))


def validation_score(token_summary,motion_summary):return (motion_summary["collapse_rate"],token_summary["normalized_token_edit_distance"],-token_summary["micro_token_accuracy"],motion_summary["normalized_feature_mae"])


def full_candidate_evaluation(state,label):
    model=fresh_model();load_state_into(model,state);mapping=exact_generate(model,val_frame,label=label+" full validation")
    token_summary,_=token_metrics(val_frame,mapping,label);motion_summary,_=motion_metrics(val_frame,mapping,label)
    del model,mapping;gc.collect();torch.cuda.empty_cache();return token_summary,motion_summary


variant_specs={"control":("control",0.),"sc_point":("point",selected_p["point"]),"sc_span":("span",selected_p["span"])}
selected_states={};validation_results={};training_histories={}
for label,(mode,pmax) in variant_specs.items():
    model,history,candidates=train_epochs(mode,pmax,train_frame,CFG.epochs,screen=True);history.to_csv(TRAINING/f"{label}_history.csv",index=False);training_histories[label]=history
    evaluated=[]
    for candidate in candidates[:CFG.full_validation_candidates]:
        ts,ms=full_candidate_evaluation(candidate["state"],f"{label}_epoch{candidate['epoch']}");evaluated.append((validation_score(ts,ms),candidate,ts,ms))
    evaluated.sort(key=lambda x:x[0]);_,best,ts,ms=evaluated[0];selected_states[label]=best["state"];validation_results[label]={"epoch":best["epoch"],"token":ts,"motion":ms,"p_max":pmax}
    destination=MODELS/label;destination.mkdir(parents=True,exist_ok=True)
    torch.save({"motion_state_dict":best["state"],"source_phase4_sha256":sha(P4_CKPT),"selected_epoch":best["epoch"],"validation":validation_results[label]},destination/"best_model.pt")
    write_json(destination/"config.json",{"architecture_changed":False,"mode":mode,"p_max":pmax,"schedule":"two-pass linear warmup",
        "optimizer":"AdamW","learning_rate":CFG.learning_rate,"weight_decay":CFG.weight_decay,"epochs_run":CFG.epochs,
        "selected_epoch":best["epoch"],"selection_policy":selection_policy,"dependency_manifest":resolved})
    del model,candidates;gc.collect();torch.cuda.empty_cache()

original_model=fresh_model();original_mapping=exact_generate(original_model,val_frame,label="original full validation")
original_val_token,_=token_metrics(val_frame,original_mapping,"original");original_val_motion,_=motion_metrics(val_frame,original_mapping,"original")
validation_results["original"]={"epoch":0,"token":original_val_token,"motion":original_val_motion,"p_max":0.}
del original_mapping;gc.collect();torch.cuda.empty_cache()
best_self=min(("sc_point","sc_span"),key=lambda x:validation_score(validation_results[x]["token"],validation_results[x]["motion"]))
best_overall=min(("control","sc_point","sc_span"),key=lambda x:validation_score(validation_results[x]["token"],validation_results[x]["motion"]))
print("Best self-conditioned:",best_self,"Best Phase-7:",best_overall)

# %% [markdown]
# ## 7.12–7.22 — Frozen test evaluation and exposure-bias diagnostics

# %%
def save_npz(path,ids,mapping):
    arrays=[mapping[x] for x in ids];lengths=np.asarray([a.shape[1] for a in arrays],np.int32);offsets=np.concatenate([[0],np.cumsum(lengths)]).astype(np.int64)
    np.savez_compressed(path,sample_ids=np.asarray(ids,str),lengths=lengths,offsets=offsets,tokens_flat=np.concatenate(arrays,1))


model_states={"original":torch.load(P4_CKPT,map_location="cpu",weights_only=True)["motion_state_dict"],**selected_states}
test_ids=test_frame[ID].astype(str).tolist();test_token_summaries={};test_token_details={};test_motion_summaries={};test_motion_details={};teacher_results={};test_maps={}
for label,state in model_states.items():
    model=fresh_model();load_state_into(model,state);teacher_results[label]=teacher_evaluate(model,test_frame)
    mapping=exact_generate(model,test_frame,label=label+" oracle-length test");test_maps[label]=mapping
    save_npz(TOKENS/"oracle_length"/f"{label}.npz",test_ids,mapping)
    test_token_summaries[label],test_token_details[label]=token_metrics(test_frame,mapping,label)
    destination=DECODED/"oracle_length"/label
    test_motion_summaries[label],test_motion_details[label]=motion_metrics(test_frame,mapping,label,destination)
    del model;gc.collect();torch.cuda.empty_cache()

teacher_rows=[]
for label,result in teacher_results.items():teacher_rows.append({"model":label,"loss":result["loss"],"motion_only_accuracy":result["motion_only_accuracy"],**{f"level_{i}_accuracy":x for i,x in enumerate(result["per_level_accuracy"])}})
pd.DataFrame(teacher_rows).to_csv(RESULTS/"teacher_forced_comparison.csv",index=False)
token_rows=[];motion_rows=[]
for label in model_states:
    token_rows.append({**test_token_summaries[label],**{f"level_{i}_accuracy":x for i,x in enumerate(test_token_summaries[label]["per_level_accuracy"])}})
    motion_rows.append(test_motion_summaries[label])
pd.DataFrame(token_rows).to_csv(RESULTS/"oracle_length_token_comparison.csv",index=False);pd.DataFrame(motion_rows).to_csv(RESULTS/"oracle_length_motion_comparison.csv",index=False)

# Fixed Phase-6 length predictor and calibration.
length_config=read_json(P6_LENGTH_CONFIG);bounds=length_config["training_bounds"]
if list(map(int,bounds))!=[24,469]:raise RuntimeError(f"Unexpected Phase-6 length bounds {bounds}")
class LengthHead(nn.Module):
    def __init__(self):super().__init__();self.network=nn.Sequential(nn.Linear(int(encoder.config.d_model),256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
    def forward(self,x):return self.network(x).squeeze(-1)
length_head=LengthHead().to(DEVICE);length_payload=torch.load(P6_LENGTH_CKPT,map_location="cpu",weights_only=True);length_head.load_state_dict(length_payload["length_head_state_dict"],strict=True);length_head.eval()
for p in length_head.parameters():p.requires_grad=False
calibration=length_config["calibration"]


@torch.inference_mode()
def predict_lengths(texts):
    values=[]
    for start in range(0,len(texts),64):
        encoded=tokenizer(texts[start:start+64],padding=True,truncation=False,return_tensors="pt");ids=encoded["input_ids"].to(DEVICE);mask=encoded["attention_mask"].to(DEVICE)
        hidden=encoder(input_ids=ids,attention_mask=mask).last_hidden_state.float();pooled=(hidden*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        raw=torch.expm1(length_head(pooled)).cpu().numpy();values.extend(raw.tolist())
    values=np.asarray(values)
    if calibration.get("selected"):values=float(calibration["slope"])*values+float(calibration["intercept"])
    return np.clip(np.rint(values),int(bounds[0]),int(bounds[1])).astype(int)


phase3_test=pd.read_csv(P3/"results/test_predictions.csv",dtype={"sample_id":str});end_frame=test_frame.merge(phase3_test[["sample_id","predicted_gloss"]],left_on=ID,right_on="sample_id",validate="one_to_one")
gt_lengths=predict_lengths(test_frame[GLOSS].astype(str).tolist());end_lengths=predict_lengths(end_frame.predicted_gloss.astype(str).tolist())
best_model=fresh_model();load_state_into(best_model,selected_states[best_overall])
pred_gt_map=exact_generate(best_model,test_frame,GLOSS,gt_lengths,"best predicted-length GT-gloss")
pred_end_map=exact_generate(best_model,end_frame,"predicted_gloss",end_lengths,"best predicted-length end-to-end")
save_npz(TOKENS/"predicted_length/best_oracle_gloss.npz",test_ids,pred_gt_map);save_npz(TOKENS/"predicted_length/best_end2end.npz",test_ids,pred_end_map)
predicted_rows=[]
for label,mapping in (("best_oracle_gloss",pred_gt_map),("best_end2end",pred_end_map)):
    ts,_=token_metrics(test_frame,mapping,label);ms,_=motion_metrics(test_frame,mapping,label,DECODED/"predicted_length"/label)
    predicted_rows.append({**ts,**ms})
pd.DataFrame(predicted_rows).to_csv(RESULTS/"predicted_length_comparison.csv",index=False)

# Exposure-bias gap uses comparable micro token accuracies.
gap_rows=[]
for label in model_states:gap_rows.append({"model":label,"teacher_forced_motion_accuracy":teacher_results[label]["motion_only_accuracy"],
    "autoregressive_micro_token_accuracy":test_token_summaries[label]["micro_token_accuracy"],
    "exposure_bias_gap":teacher_results[label]["motion_only_accuracy"]-test_token_summaries[label]["micro_token_accuracy"]})
gap_frame=pd.DataFrame(gap_rows);gap_frame.to_csv(RESULTS/"exposure_bias_gap.csv",index=False)

# Accuracy over normalized temporal progress.
progress_rows=[]
diagnostic_models=list(dict.fromkeys(["original",best_overall]))
for label in diagnostic_models:
    counts=np.zeros((10,6));correct=np.zeros((10,6))
    for sid in test_ids:
        gt=tokens_for(sid);pred=test_maps[label][sid];n=min(gt.shape[1],pred.shape[1])
        for t in range(n):
            b=min(9,int(10*t/max(gt.shape[1],1)));counts[b]+=1;correct[b]+=(gt[:,t]==pred[:,t])
    for b in range(10):progress_rows.append({"model":label,"progress_bin":f"{b*10}-{(b+1)*10}%","overall_accuracy":float(correct[b].sum()/max(counts[b].sum(),1)),**{f"level_{l}_accuracy":float(correct[b,l]/max(counts[b,l],1)) for l in range(6)}})
progress=pd.DataFrame(progress_rows);progress.to_csv(RESULTS/"accuracy_vs_sequence_progress.csv",index=False)

# First-error propagation.
first_error_rows=[]
for label in diagnostic_models:
    for sid in test_ids:
        gt,pred=tokens_for(sid),test_maps[label][sid];n=min(gt.shape[1],pred.shape[1]);mismatch=np.flatnonzero(~(gt[:,:n]==pred[:,:n]).all(0));first=int(mismatch[0]) if len(mismatch) else n
        row={"model":label,"sample_id":sid,"first_error_step":first}
        for window in (5,10,20):
            start=min(n,first+1);stop=min(n,start+window);row[f"next_{window}_accuracy"]=float((gt[:,start:stop]==pred[:,start:stop]).mean()) if stop>start else float("nan")
        first_error_rows.append(row)
first_error=pd.DataFrame(first_error_rows);first_error.to_csv(RESULTS/"first_error_propagation.csv",index=False)


def repeated_ngram_rate(tokens,n=3):
    vectors=[tuple(tokens[:,i]) for i in range(tokens.shape[1])];grams=[tuple(vectors[i:i+n]) for i in range(max(0,len(vectors)-n+1))]
    return 1-len(set(grams))/max(len(grams),1)
repetition_rows=[]
for label in diagnostic_models:
    motion_detail=test_motion_details[label].set_index("sample_id")
    for sid in test_ids:
        values=test_maps[label][sid];duplicates=float(np.mean(np.all(values[:,1:]==values[:,:-1],axis=0))) if values.shape[1]>1 else 0
        repetition_rows.append({"model":label,"sample_id":sid,"repeated_trigram_rate":repeated_ngram_rate(values),
            "consecutive_duplicate_step_rate":duplicates,"motion_repetition_score":motion_detail.loc[sid,"repetition_score"],"collapse":motion_detail.loc[sid,"collapse"]})
repetition_frame=pd.DataFrame(repetition_rows);repetition_frame.to_csv(RESULTS/"repetition_analysis.csv",index=False)

# Deterministic corrupted-history robustness: predict targets after corrupted history positions.
@torch.inference_mode()
def robustness(model,label,rates=(.05,.10,.20)):
    data,_=loader(test_frame,shuffle=False,seed=71);rows=[];model.eval()
    for rate in rates:
        correct=total=0
        for batch_number,raw in enumerate(data):
            batch=move(raw);valid=~batch["padding"][:,1:];g=torch.Generator().manual_seed(CFG.seed+batch_number+round(rate*1000));selected=point_mask(valid,rate,g)
            corrupted=batch["decoder"].clone();wrong=(corrupted[:,:,1:]+1)%512;corrupted[:,:,1:]=torch.where(selected[:,None,:],wrong,corrupted[:,:,1:])
            with torch.autocast("cuda",dtype=AMP_DTYPE):pred=model(batch["ids"],batch["mask"],corrupted,batch["padding"]).argmax(-1)
            affected=torch.zeros_like(batch["padding"]);affected[:,1:]=selected;mask=(batch["targets"]<512)&affected[:,None,:]
            correct+=int((pred[mask]==batch["targets"][mask]).sum());total+=int(mask.sum())
        rows.append({"model":label,"corruption_rate":rate,"subsequent_token_accuracy":correct/max(total,1),"evaluated_tokens":total})
    return rows
robust_rows=[]
for label in ("original","control",best_self):
    model=fresh_model();load_state_into(model,model_states[label]);robust_rows.extend(robustness(model,label));del model;gc.collect();torch.cuda.empty_cache()
robustness_frame=pd.DataFrame(robust_rows);robustness_frame.to_csv(RESULTS/"history_robustness.csv",index=False)

# %% [markdown]
# ## 7.23–7.31 — Comparisons, causal decision, plots, report, and Phase-8 handoff

# %%
main_rows=[]
for label in model_states:
    ts,ms=test_token_summaries[label],test_motion_summaries[label];tf=teacher_results[label]
    main_rows.append({"model":label,"teacher_forced_accuracy":tf["motion_only_accuracy"],"aligned_token_accuracy":ts["aligned_token_accuracy"],
        "micro_token_accuracy":ts["micro_token_accuracy"],"normalized_token_edit_distance":ts["normalized_token_edit_distance"],
        **{name:ts["per_level_accuracy"][i] for i,name in enumerate(["base_level_accuracy","residual_1_accuracy","residual_2_accuracy","residual_3_accuracy","residual_4_accuracy","residual_5_accuracy"])},
        "normalized_feature_mae":ms["normalized_feature_mae"],"velocity_mae":ms["velocity_mae"],"motion_variance":ms["motion_variance"],
        "collapse_rate":ms["collapse_rate"],"repetition_score":ms["repetition_score"],"extreme_motion_rate":ms["extreme_motion_rate"]})
main=pd.DataFrame(main_rows);main.to_csv(RESULTS/"phase7_main_comparison.csv",index=False)

original=main.set_index("model").loc["original"];best=main.set_index("model").loc[best_overall];control=main.set_index("model").loc["control"]
progress_pivot=progress.pivot(index="progress_bin",columns="model",values="overall_accuracy")
late_improved=False if best_overall=="original" else float(progress_pivot[best_overall].iloc[-3:].mean())>float(progress_pivot["original"].iloc[-3:].mean())
first_summary=first_error.groupby("model")[["next_5_accuracy","next_10_accuracy","next_20_accuracy"]].mean()
first_improved=False if best_overall=="original" else float(first_summary.loc[best_overall].mean())>float(first_summary.loc["original"].mean())
robust_summary=robustness_frame.groupby("model").subsequent_token_accuracy.mean();robust_improved=float(robust_summary[best_self])>float(robust_summary["original"])
teacher_deterioration=float(teacher_results[best_overall]["motion_only_accuracy"]-teacher_results["original"]["motion_only_accuracy"])
self_beats_control=validation_score(validation_results[best_self]["token"],validation_results[best_self]["motion"])<validation_score(validation_results["control"]["token"],validation_results["control"]["motion"])
base_gain=float(best.base_level_accuracy-original.base_level_accuracy);residual_gain=float(np.mean([best[f"residual_{i}_accuracy"]-original[f"residual_{i}_accuracy"] for i in range(1,6)]))
meaningful=(float(best.collapse_rate)<float(original.collapse_rate)-.02 or float(best.normalized_token_edit_distance)<float(original.normalized_token_edit_distance)-.01) and float(best.normalized_feature_mae)<=float(original.normalized_feature_mae)+.03
if meaningful:dominant="EXPOSURE_BIAS_REDUCED";recommendation="refine the best exposure-aware model; then test repetition/dynamics regularization"
elif base_gain>residual_gain+.02:dominant="RESIDUAL_RVQ_LEVELS";recommendation="investigate hierarchical coarse-to-fine RVQ prediction"
else:dominant="TEMPORAL_GENERATION_ARCHITECTURE";recommendation="investigate non-autoregressive or masked temporal generation"
causal={"teacher_forcing_control_helped":bool(validation_score(validation_results["control"]["token"],validation_results["control"]["motion"])<validation_score(validation_results["original"]["token"],validation_results["original"]["motion"])),
    "self_conditioning_outperformed_control":self_beats_control,"better_self_conditioning_variant":best_self,
    "token_accuracy_change":float(best.micro_token_accuracy-original.micro_token_accuracy),
    "edit_distance_change":float(best.normalized_token_edit_distance-original.normalized_token_edit_distance),
    "collapse_change":float(best.collapse_rate-original.collapse_rate),"repetition_change":float(best.repetition_score-original.repetition_score),
    "late_sequence_accuracy_improved":late_improved,"first_error_propagation_improved":first_improved,
    "history_robustness_improved":robust_improved,"teacher_forced_accuracy_change":teacher_deterioration,
    "dominant_remaining_failure":dominant,"recommended_phase8":recommendation}
write_json(RESULTS/"phase7_causal_analysis.json",causal)

# Required plots.
fig,axes=plt.subplots(1,3,figsize=(15,4))
for label,history in training_histories.items():axes[0].plot(history.epoch,history.train_loss,label=label);axes[1].plot(history.epoch,history.validation_motion_accuracy,label=label);axes[2].plot(history.epoch,history.p_corruption,label=label)
for ax in axes:ax.legend();ax.grid(alpha=.2)
axes[0].set_title("Training loss");axes[1].set_title("Teacher-forced validation accuracy");axes[2].set_title("Corruption schedule")
fig.tight_layout();fig.savefig(FIGURES/"training_curves.png",dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for label in diagnostic_models:
    part=progress[progress.model==label];ax.plot(range(10),part.overall_accuracy,marker="o",label=label)
ax.set(xticks=range(10),xticklabels=[f"{i*10}-{(i+1)*10}%" for i in range(10)],xlabel="Sequence progress",ylabel="Token accuracy");ax.tick_params(axis="x",rotation=35);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"token_accuracy_vs_sequence_progress.png",dpi=170);plt.close(fig)
for column,filename,title in (("collapse_rate","collapse_comparison.png","Collapse rate"),("repetition_score","repetition_comparison.png","Repetition score")):
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(main.model,main[column]);ax.set_title(title);ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(FIGURES/filename,dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for label,part in robustness_frame.groupby("model"):ax.plot(part.corruption_rate,part.subsequent_token_accuracy,marker="o",label=label)
ax.set(xlabel="History corruption rate",ylabel="Affected-position accuracy",title="Corrupted-history robustness");ax.legend();fig.tight_layout();fig.savefig(FIGURES/"history_robustness.png",dpi=170);plt.close(fig)

predicted_comparison=pd.read_csv(RESULTS/"predicted_length_comparison.csv");pred_end=predicted_comparison.query("model=='best_end2end'").iloc[0].to_dict()
phase7_complete=bool(len(main)==4 and all((MODELS/x/"best_model.pt").is_file() for x in ("control","sc_point","sc_span")) and len(robustness_frame)>0)
report={"phase":"Phase 7 - Autoregressive Exposure-Bias Repair","phase7_complete":phase7_complete,"phase1_fingerprint":FINGERPRINT,
    "initial_checkpoint":str(P4_CKPT),"initial_checkpoint_sha256":sha(P4_CKPT),"architecture_changed":False,"length_model_changed":False,
    "models":{label:{"teacher_forced":teacher_results[label],"oracle_length_tokens":test_token_summaries[label],"oracle_length_motion":test_motion_summaries[label],"validation":validation_results[label]} for label in model_states},
    "best_model":best_overall,"best_self_conditioned_model":best_self,"selected_corruption_probabilities":selected_p,
    "oracle_length_test":{**test_token_summaries[best_overall],**test_motion_summaries[best_overall]},
    "predicted_length_end2end":pred_end,"exposure_bias":{"original_gap":float(gap_frame.set_index("model").loc["original","exposure_bias_gap"]),
        "best_phase7_gap":float(gap_frame.set_index("model").loc[best_overall,"exposure_bias_gap"]),"first_error_propagation_improved":first_improved,"history_robustness_improved":robust_improved},
    "dominant_remaining_failure":dominant,"recommended_phase8":recommendation,"ready_for_phase8":phase7_complete}
write_json(OUT/"phase7_config.json",{"config":asdict(CFG),"resolved":resolved,"selection_policy":selection_policy})
write_json(OUT/"phase7_report.json",report)

readme=f"""PHASE 8 HANDOFF — MOTION-S / SIGNVRSE

Best Phase-7 model: {best_overall}
Best self-conditioned variant: {best_self}; selected probabilities: {selected_p}
Intervention: two-pass self-conditioned history corruption (not exact scheduled sampling).
Architecture changed: NO. Phase-3 changed: NO. Phase-6 length predictor changed: NO.

Oracle-length original -> best:
micro token accuracy {original.micro_token_accuracy:.6f} -> {best.micro_token_accuracy:.6f}
normalized edit distance {original.normalized_token_edit_distance:.6f} -> {best.normalized_token_edit_distance:.6f}
normalized feature MAE {original.normalized_feature_mae:.6f} -> {best.normalized_feature_mae:.6f}
collapse {original.collapse_rate:.6f} -> {best.collapse_rate:.6f}
repetition {original.repetition_score:.6f} -> {best.repetition_score:.6f}

Late-sequence accuracy improved: {late_improved}
First-error propagation improved: {first_improved}
Corrupted-history robustness improved: {robust_improved}
Remaining weakness: {dominant}
Recommended Phase 8: {recommendation}

No skeleton/BVH conversion, MPJPE, or semantic sign-quality claim is supported.
"""
(OUT/"README_phase8.txt").write_text(readme,encoding="utf-8")
inventory=[]
for path in sorted(OUT.rglob("*")):
    if path.is_file() and path.name!="artifact_inventory.csv":inventory.append({"relative_path":str(path.relative_to(OUT)),"bytes":path.stat().st_size,"sha256":sha(path)})
pd.DataFrame(inventory).to_csv(OUT/"artifact_inventory.csv",index=False)

def show(label):
    row=main.set_index("model").loc[label];print(label);print("Aligned token accuracy:",f"{row.aligned_token_accuracy:.6f}");print("Normalized edit distance:",f"{row.normalized_token_edit_distance:.6f}");print("Feature MAE:",f"{row.normalized_feature_mae:.6f}");print("Collapse:",f"{row.collapse_rate:.6f}");print("Repetition:",f"{row.repetition_score:.6f}")
print("="*60);print("PHASE 7 COMPLETE");print("="*60);show("original");print("-"*60);show("control");print("-"*60);print("SC-POINT selected p_max",selected_p["point"]);show("sc_point");print("-"*60);print("SC-SPAN selected p_max",selected_p["span"]);show("sc_span")
print("-"*60);print("BEST PHASE-7 MODEL",best_overall);print("Exposure-bias gap reduction:",float(gap_frame.set_index("model").loc["original","exposure_bias_gap"]-gap_frame.set_index("model").loc[best_overall,"exposure_bias_gap"]));print("First-error propagation:","IMPROVED" if first_improved else "NOT IMPROVED");print("History robustness:","IMPROVED" if robust_improved else "NOT IMPROVED");print("Predicted-length end-to-end collapse:",pred_end["collapse_rate"]);print("Predicted-length end-to-end feature MAE:",pred_end["normalized_feature_mae"]);print("Dominant remaining bottleneck:",dominant);print("Recommended Phase 8:",recommendation);print("READY FOR PHASE 8:","YES" if phase7_complete else "NO");print("="*60)
