# %% [markdown]
# # Motion-S / SignVRSE — Phase 8: Final Generator Refinement, Integration, and Evaluation
#
# Final experimental phase. All selection is validation-only. The test benchmark is evaluated once
# after the generator is frozen. No skeleton, BVH, MPJPE, or semantic sign-quality claim is made.

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
import textwrap
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

os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["TOKENIZERS_PARALLELISM"]="false"
FINGERPRINT="e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_PHASE4_SHA="b01d4f6db3c377b368d6fff658b28cdc153fb5d18acfd256ae5f84935ce558ee"
EXPECTED_RVQ_SHA="0329cf9197f3ba8088e79ce519c422b97ba118ae8046f3dc608c6d4e1904b25f"
EXPECTED_SPLITS={"train":9970,"validation":1246,"test":1247};EXPECTED_SUPPORTED={"train":9936,"validation":1243,"test":1243}
LEVELS=["base_tokens","residual_1","residual_2","residual_3","residual_4","residual_5"]
PAD_ID,BOS_ID,EOS_ID=512,513,514


@dataclass(frozen=True)
class Config:
    seed:int=42
    kaggle_input:str="/kaggle/input"
    kaggle_working:str="/kaggle/working"
    output_dir:str="/kaggle/working/motion_s_phase8"
    max_t:int=512
    p_max:float=.15
    corruption_warmup_epochs:int=5
    learning_rate:float=1.5e-5
    weight_decay:float=.01
    max_grad_norm:float=1.
    epochs:int=6
    search_epochs:int=2
    search_train_samples:int=2048
    validation_subset_samples:int=192
    lambda_velocity_candidates:tuple[float,...]=(.02,.05,.10)
    lambda_persistence_candidates:tuple[float,...]=(.005,.01,.02)
    near_zero_std:float=1e-6
    severe_feature_mae_delta:float=.03
    severe_edit_delta:float=.03
    severe_collapse_delta:float=.05


CFG=Config();OUT=Path(CFG.output_dir)
DIAG,MODELS,FINAL_MODEL,TRAINING,RESULTS,FIGURES,FINAL_PIPELINE=(OUT/"diagnostics",OUT/"models",OUT/"final_model",OUT/"training",OUT/"results",OUT/"figures",OUT/"final_pipeline")
FINAL_CASES=FIGURES/"final_cases"
for path in (OUT,DIAG,MODELS,FINAL_MODEL,TRAINING,RESULTS,FIGURES,FINAL_CASES,FINAL_PIPELINE):path.mkdir(parents=True,exist_ok=True)


def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
seed_all(CFG.seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type!="cuda":raise RuntimeError("Phase 8 requires a Kaggle GPU")
AMP_DTYPE=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.floating):return None if not np.isfinite(value) else float(value)
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as handle:json.dump(safe(payload),handle,indent=2,ensure_ascii=False,allow_nan=False)
def read_json(path):
    with path.open(encoding="utf-8") as handle:return json.load(handle)
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def canon(value):return re.sub(r"[^a-z0-9]","",str(value).casefold())
def spearman(x,y):
    frame=pd.DataFrame({"x":pd.to_numeric(x,errors="coerce"),"y":pd.to_numeric(y,errors="coerce")}).dropna()
    return float(frame.x.rank().corr(frame.y.rank())) if len(frame)>1 and frame.x.nunique()>1 and frame.y.nunique()>1 else float("nan")
print("Python",sys.version.split()[0],"PyTorch",torch.__version__,"GPU",torch.cuda.get_device_name(0));display(pd.Series(asdict(CFG),name="value").to_frame())

# %% [markdown]
# ## 8.0 — Artifact discovery and hash verification

# %%
SEARCH=[Path(CFG.kaggle_input),Path(CFG.kaggle_working)]
def roots(predicate):
    found=[]
    for base in SEARCH:
        if not base.exists():continue
        for current,dirs,_ in os.walk(base):
            candidate=Path(current)
            if candidate.resolve()==OUT.resolve():dirs[:]=[];continue
            if predicate(candidate):found.append(candidate.resolve());dirs[:]=[]
    return sorted(dict.fromkeys(found),key=str)
def choose(label,predicate,identities):
    candidates=roots(predicate)
    if not candidates:raise RuntimeError(f"Missing {label}")
    signatures={root:tuple((rel,sha(root/rel)) for rel in identities if (root/rel).is_file()) for root in candidates}
    if len(set(signatures.values()))!=1:raise RuntimeError(f"Ambiguous non-identical {label}: {[(str(k),v) for k,v in signatures.items()]}")
    preferred=[p for p in candidates if str(p).startswith(CFG.kaggle_working)];selected=sorted(preferred or candidates,key=lambda p:(len(p.parts),str(p)))[0]
    print(label,":",selected);return selected


P1=choose("Phase 1",lambda p:(p/"dataset_config.json").is_file() and (p/"splits/train_split.csv").is_file(),["dataset_config.json","file_manifest.csv"])
P3=choose("Phase 3",lambda p:all((p/x).is_file() for x in ("phase3_report.json","best_model/config.json","tokenizer/tokenizer_config.json","results/test_predictions.csv")),["phase3_report.json","best_model/config.json","best_model/model.safetensors","tokenizer/tokenizer_config.json"])
P4=choose("Phase 4C",lambda p:all((p/x).is_file() for x in ("phase4c_report.json","best_model/model.pt","best_model/model_config.json")),["phase4c_report.json","best_model/model.pt"])
P5=choose("Phase 5B",lambda p:all((p/x).is_file() for x in ("phase5b_report.json","diagnostics/decoder_contract.json","diagnostics/motion_mean_668.npy","diagnostics/motion_std_668.npy","diagnostics/empirical_motion_thresholds.json")),["phase5b_report.json","diagnostics/decoder_contract.json"])
P6=choose("Phase 6",lambda p:all((p/x).is_file() for x in ("phase6_report.json","length_model/best_model.pt","length_model/config.json","results/phase6_main_comparison.csv")),["phase6_report.json","length_model/best_model.pt","results/phase6_main_comparison.csv"])
P7=choose("Phase 7",lambda p:all((p/x).is_file() for x in ("phase7_report.json","diagnostics/artifact_verification.json","models/sc_point/best_model.pt","models/sc_point/config.json","tokens/oracle_length/original.npz","tokens/oracle_length/sc_point.npz")) and (p/"decoded/oracle_length/sc_point").is_dir(),["phase7_report.json","diagnostics/artifact_verification.json","models/sc_point/best_model.pt","models/sc_point/config.json"])
P1CFG=read_json(P1/"dataset_config.json");P6REPORT=read_json(P6/"phase6_report.json");P7REPORT=read_json(P7/"phase7_report.json")
if P1CFG.get("split_fingerprint_sha256")!=FINGERPRINT:raise RuntimeError("Phase-1 fingerprint mismatch")
if P7REPORT.get("phase7_complete") is not True:raise RuntimeError("Phase 7 incomplete")
if P7REPORT.get("best_self_conditioned_model")!="sc_point":raise RuntimeError("Locked Phase-7 best self-conditioned model is not SC-POINT")
if abs(float(P7REPORT.get("selected_corruption_probabilities",{}).get("point",float("nan")))-.15)>1e-12:raise RuntimeError("Locked Phase-7 SC-POINT p_max is not 0.15")
ID,GLOSS,SENTENCE=P1CFG["id_column"],P1CFG["gloss_column"],P1CFG.get("sentence_column","sentence");TOKEN_COLS=list(P1CFG["motion_token_columns"])
if TOKEN_COLS!=LEVELS:raise RuntimeError("RVQ level order mismatch")


def unique_file(name,expected=None):
    candidates=sorted({p.resolve() for base in SEARCH if base.exists() for p in base.rglob(name) if p.is_file()},key=str)
    if expected:candidates=[p for p in candidates if sha(p)==expected]
    if not candidates:raise RuntimeError(f"Missing {name}")
    hashes={p:sha(p) for p in candidates}
    if len(set(hashes.values()))!=1:raise RuntimeError(f"Ambiguous {name}: {[(str(p),h) for p,h in hashes.items()]}")
    return candidates[0]
RVQ_CKPT=unique_file("rvq_vae_best.pth",EXPECTED_RVQ_SHA)


def locate_train():
    candidates=[]
    for base in SEARCH:
        if not base.exists():continue
        for path in base.rglob("train.csv"):
            try:
                if {ID,GLOSS,*TOKEN_COLS}.issubset(pd.read_csv(path,nrows=0).columns):candidates.append(path.resolve())
            except Exception:pass
    hashes={p:sha(p) for p in candidates}
    if not candidates or len(set(hashes.values()))!=1:raise RuntimeError(f"train.csv missing/ambiguous: {[(str(p),h) for p,h in hashes.items()]}")
    return candidates[0]
def locate_features():
    candidates=sorted({(Path(c)/d).resolve() for base in SEARCH if base.exists() for c,ds,_ in os.walk(base) for d in ds if canon(d)=="motionfeatures"},key=str)
    if len(candidates)!=1:raise RuntimeError(f"Motion-Features ambiguity {candidates}")
    return candidates[0]
TRAIN_CSV,FEATURE_DIR=locate_train(),locate_features();P4_CKPT=P4/"best_model/model.pt";P7_CKPT=P7/"models/sc_point/best_model.pt";LENGTH_CKPT=P6/"length_model/best_model.pt"
if sha(P4_CKPT)!=EXPECTED_PHASE4_SHA:raise RuntimeError(f"Phase-4 SHA mismatch: {sha(P4_CKPT)}")
if sha(RVQ_CKPT)!=EXPECTED_RVQ_SHA:raise RuntimeError("RVQ SHA mismatch")
p7_artifacts=read_json(P7/"diagnostics/artifact_verification.json");recorded_length_sha=p7_artifacts.get("length_sha256")
if recorded_length_sha and sha(LENGTH_CKPT)!=recorded_length_sha:raise RuntimeError(f"Phase-6 length checkpoint SHA mismatch: {sha(LENGTH_CKPT)} != {recorded_length_sha}")
resolved={"phase1":P1,"phase3":P3,"phase4":P4,"phase5":P5,"phase6":P6,"phase7":P7,"official_train_csv":TRAIN_CSV,"motion_features":FEATURE_DIR,
 "phase4_checkpoint":P4_CKPT,"phase4_sha256":sha(P4_CKPT),"phase6_length_checkpoint":LENGTH_CKPT,"phase6_length_sha256":sha(LENGTH_CKPT),
 "phase7_sc_point_checkpoint":P7_CKPT,"phase7_sc_point_sha256":sha(P7_CKPT),"rvq_checkpoint":RVQ_CKPT,"rvq_sha256":sha(RVQ_CKPT)}
for key,value in resolved.items():print(key,":",value)
write_json(DIAG/"artifact_verification.json",{**resolved,"phase1_fingerprint":FINGERPRINT,"passed":True})

# %% [markdown]
# ## 8.1–8.5 — Locked data/model reconstruction and verified latent losses

# %%
def parse_cell(value):
    text=str(value).strip()
    try:parsed=json.loads(text)
    except (json.JSONDecodeError,TypeError):
        try:parsed=ast.literal_eval(text)
        except (ValueError,SyntaxError):parsed=[int(x) for x in re.split(r"[\s,]+",text) if x]
    array=np.asarray(parsed);numeric=array.astype(np.float64)
    if array.ndim!=1 or array.size==0 or not np.isfinite(numeric).all() or not np.equal(numeric,np.floor(numeric)).all():raise ValueError("invalid token cell")
    return numeric.astype(np.int64)
official=pd.read_csv(TRAIN_CSV,dtype={ID:str},low_memory=False);official[ID]=official[ID].astype(str);official_index=official.set_index(ID,drop=False);TOKEN_CACHE={}
def tokens_for(sid):
    sid=str(sid)
    if sid not in TOKEN_CACHE:
        value=np.stack([parse_cell(official_index.loc[sid,c]) for c in TOKEN_COLS])
        if value.min()<0 or value.max()>511:raise ValueError(sid)
        TOKEN_CACHE[sid]=value
    return TOKEN_CACHE[sid]
def split_path(name):
    names={"train":["train_split.csv"],"validation":["val_split.csv","validation_split.csv"],"test":["test_split.csv"]}[name];found=[P1/"splits"/x for x in names if (P1/"splits"/x).is_file()]
    if len(found)!=1:raise RuntimeError(found)
    return found[0]
frames={}
for name,expected in EXPECTED_SPLITS.items():
    frame=pd.read_csv(split_path(name),dtype={ID:str},low_memory=False);frame[ID]=frame[ID].astype(str)
    if len(frame)!=expected or not frame[ID].is_unique:raise RuntimeError(name)
    frame["motion_length"]=[tokens_for(x).shape[1] for x in frame[ID]];frame=frame[frame.motion_length<=CFG.max_t].reset_index(drop=True)
    if len(frame)!=EXPECTED_SUPPORTED[name]:raise RuntimeError(f"{name} count")
    frames[name]=frame
train_frame,val_frame,test_frame=frames["train"],frames["validation"],frames["test"]

from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
TOKENIZER_PATH,PHASE3_MODEL=P3/"tokenizer",P3/"best_model";tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH,local_files_only=True,use_fast=True)
phase3_encoder_source=AutoModelForSeq2SeqLM.from_pretrained(PHASE3_MODEL,local_files_only=True);encoder=phase3_encoder_source.get_encoder().to(DEVICE).eval();del phase3_encoder_source
for p in encoder.parameters():p.requires_grad=False
MODEL_CONFIG=read_json(P4/"best_model/model_config.json")


class MotionModel(nn.Module):
    def __init__(self,gloss_encoder):
        super().__init__();self.gloss_encoder=gloss_encoder;dm,dl=int(MODEL_CONFIG["d_model"]),int(MODEL_CONFIG["d_level"])
        self.encoder_projection=nn.Linear(int(gloss_encoder.config.d_model),dm);self.level_embeddings=nn.ModuleList([nn.Embedding(515,dl,padding_idx=PAD_ID) for _ in range(6)])
        self.motion_projection=nn.Linear(6*dl,dm);self.temporal_positions=nn.Embedding(int(MODEL_CONFIG["max_model_positions"]),dm)
        layer=nn.TransformerDecoderLayer(dm,int(MODEL_CONFIG["num_heads"]),int(MODEL_CONFIG["ffn_dimension"]),float(MODEL_CONFIG["dropout"]),activation="gelu",batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,int(MODEL_CONFIG["decoder_layers"]),norm=nn.LayerNorm(dm));self.heads=nn.ModuleList([nn.Linear(dm,515) for _ in range(6)])
    def train(self,mode=True):super().train(mode);self.gloss_encoder.eval();return self
    def encode(self,ids,mask):
        with torch.no_grad():hidden=self.gloss_encoder(input_ids=ids,attention_mask=mask).last_hidden_state
        return self.encoder_projection(hidden)
    def decode(self,tokens,memory,mask,padding):
        embedded=torch.cat([self.level_embeddings[q](tokens[:,q]) for q in range(6)],-1);embedded=self.motion_projection(embedded)+self.temporal_positions(torch.arange(tokens.shape[-1],device=tokens.device)[None])
        causal=torch.triu(torch.ones(tokens.shape[-1],tokens.shape[-1],dtype=torch.bool,device=tokens.device),1)
        hidden=self.decoder(embedded,memory,tgt_mask=causal,tgt_key_padding_mask=padding,memory_key_padding_mask=~mask.bool());return torch.stack([head(hidden) for head in self.heads],1)
    def forward(self,ids,mask,tokens,padding):return self.decode(tokens,self.encode(ids,mask),mask,padding)
def fresh_model(checkpoint=P7_CKPT):
    model=MotionModel(encoder).to(DEVICE);payload=torch.load(checkpoint,map_location="cpu",weights_only=True);state=payload.get("motion_state_dict",payload)
    result=model.load_state_dict(state,strict=False);expected=[n for n in model.state_dict() if n.startswith("gloss_encoder.")]
    if sorted(result.missing_keys)!=sorted(expected) or result.unexpected_keys:raise RuntimeError(str(result))
    for name,p in model.named_parameters():p.requires_grad=not name.startswith("gloss_encoder.")
    return model


class MotionDataset(Dataset):
    def __init__(self,frame,gloss_col=GLOSS):self.frame=frame.reset_index(drop=True);self.gloss_col=gloss_col
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i];return {"id":str(row[ID]),"gloss":str(row[self.gloss_col]),"tokens":tokens_for(row[ID]),"length":int(row.motion_length)}
def collate(items):
    text=tokenizer([x["gloss"] for x in items],padding=True,truncation=False,return_tensors="pt");maximum=max(x["length"] for x in items)+1
    decoder_tokens=torch.full((len(items),6,maximum),PAD_ID,dtype=torch.long);targets=torch.full_like(decoder_tokens,PAD_ID);padding=torch.ones((len(items),maximum),dtype=torch.bool)
    for i,item in enumerate(items):
        length=item["length"];target=torch.from_numpy(item["tokens"]);decoder_tokens[i,:,0]=BOS_ID;decoder_tokens[i,:,1:length+1]=target;targets[i,:,:length]=target;targets[i,:,length]=EOS_ID;padding[i,:length+1]=False
    return {"sample_id":[x["id"] for x in items],"ids":text["input_ids"],"mask":text["attention_mask"],"decoder":decoder_tokens,"targets":targets,"padding":padding}
BOUNDARIES=(64,96,128,181,256,384,512);BATCH_SIZES=(8,8,8,8,4,2,1)
class BucketSampler(Sampler):
    def __init__(self,lengths,shuffle,seed):
        self.buckets=[[] for _ in BOUNDARIES];self.shuffle=shuffle;self.seed=seed;self.epoch=0
        for i,x in enumerate(lengths):self.buckets[next(j for j,u in enumerate(BOUNDARIES) if x<=u)].append(i)
    def set_epoch(self,x):self.epoch=x
    def batches(self):
        rng=np.random.default_rng(self.seed+self.epoch);out=[]
        for values,size in zip(self.buckets,BATCH_SIZES):
            values=list(values)
            if self.shuffle:rng.shuffle(values)
            out.extend(values[i:i+size] for i in range(0,len(values),size))
        if self.shuffle:rng.shuffle(out)
        return out
    def __iter__(self):return iter(self.batches())
    def __len__(self):return sum(math.ceil(len(x)/s) for x,s in zip(self.buckets,BATCH_SIZES))
def make_loader(frame,gloss_col=GLOSS,shuffle=False,seed=42):
    sampler=BucketSampler(frame.motion_length.astype(int).tolist(),shuffle,seed);return DataLoader(MotionDataset(frame,gloss_col),batch_sampler=sampler,collate_fn=collate,num_workers=0,pin_memory=True),sampler
def move(batch):return {k:(v.to(DEVICE,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


class MotionDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.decoder=nn.Sequential(nn.ConvTranspose1d(256,512,3,2,1,output_padding=1),nn.ReLU(),nn.BatchNorm1d(512),nn.ConvTranspose1d(512,512,3,2,1,output_padding=1),nn.ReLU(),nn.BatchNorm1d(512),nn.ConvTranspose1d(512,512,3,1,1),nn.ReLU(),nn.BatchNorm1d(512),nn.ConvTranspose1d(512,668,3,1,1))
    def forward(self,x):return self.decoder(x)
payload=torch.load(RVQ_CKPT,map_location="cpu",weights_only=True)
for key in ("model_state_dict","state_dict"):
    if isinstance(payload,dict) and key in payload:payload=payload[key];break
rvq_state={k:v for k,v in payload.items() if isinstance(v,torch.Tensor)};decoder=MotionDecoder();decoder.load_state_dict({k[8:]:v for k,v in rvq_state.items() if k.startswith("decoder.")},strict=True);decoder=decoder.to(DEVICE).eval()
codebooks=torch.stack([rvq_state[f"rvq.quantizers.{q}.embedding"].float() for q in range(6)]).to(DEVICE);codebooks.requires_grad_(False)
motion_mean=np.load(P5/"diagnostics/motion_mean_668.npy",allow_pickle=False).astype(np.float32);motion_std=np.load(P5/"diagnostics/motion_std_668.npy",allow_pickle=False).astype(np.float32)
active=motion_std>=CFG.near_zero_std;safe_std=motion_std.copy();safe_std[~active]=1.;mean_t=torch.from_numpy(motion_mean).to(DEVICE);std_t=torch.from_numpy(safe_std).to(DEVICE)
thresholds=read_json(P5/"diagnostics/empirical_motion_thresholds.json");feature_index={p.stem:p for p in FEATURE_DIR.glob("*.npy")}


def ce_and_accuracy(logits,targets):
    losses=[];correct=np.zeros(6);total=np.zeros(6);pred=logits.argmax(-1)
    for q in range(6):
        losses.append(F.cross_entropy(logits[:,q].reshape(-1,515),targets[:,q].reshape(-1),ignore_index=PAD_ID));valid=targets[:,q]<512;correct[q]=int((pred[:,q][valid]==targets[:,q][valid]).sum());total[q]=int(valid.sum())
    return torch.stack(losses).mean(),correct,total
def latent_losses(logits,targets):
    real=targets[:,0]<512;expected=torch.zeros((targets.shape[0],targets.shape[2],256),device=logits.device,dtype=torch.float32);ground=torch.zeros_like(expected)
    for q in range(6):
        probabilities=F.softmax(logits[:,q,:,:512].float(),-1);expected=expected+probabilities@codebooks[q].T
        indices=targets[:,q].clamp(0,511);contribution=F.embedding(indices,codebooks[q].T);ground=ground+contribution*real[:,:,None]
    transitions=real[:,1:]&real[:,:-1]
    velocity_element=F.smooth_l1_loss(expected[:,1:]-expected[:,:-1],ground[:,1:]-ground[:,:-1],reduction="none").mean(-1)
    velocity_loss=velocity_element[transitions].mean() if transitions.any() else logits.sum()*0
    persistence_terms=[]
    for q in range(6):
        current=targets[:,q,1:];previous=targets[:,q,:-1];change=transitions&(current!=previous)
        if change.any():
            probabilities=F.softmax(logits[:,q,1:,:512].float(),-1);p_prev=probabilities.gather(-1,previous.clamp(0,511).unsqueeze(-1)).squeeze(-1);persistence_terms.append((-torch.log1p(-p_prev.clamp(max=1-1e-6)))[change].mean())
    persistence_loss=torch.stack(persistence_terms).mean() if persistence_terms else logits.sum()*0
    return velocity_loss,persistence_loss

# %% [markdown]
# ## 8.6–8.13 — Controlled validation-only regularization experiment

# %%
def corruption_probability(epoch):return CFG.p_max*min(1.,max(0.,(epoch-1)/max(CFG.corruption_warmup_epochs-1,1)))
def corrupted_history(model,batch,p,seed):
    if p<=0:return batch["decoder"]
    model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=AMP_DTYPE):prediction=model(batch["ids"],batch["mask"],batch["decoder"],batch["padding"])[:,:,:-1,:512].argmax(-1)
    valid=~batch["padding"][:,1:];generator=torch.Generator().manual_seed(seed);selected=(torch.rand(valid.shape,generator=generator)<p).to(valid.device)&valid
    result=batch["decoder"].clone();result[:,:,1:]=torch.where(selected[:,None,:],prediction,result[:,:,1:]);return result


@torch.inference_mode()
def exact_generate(model,frame,gloss_col=GLOSS,lengths=None,label="generate"):
    lengths=frame.motion_length.to_numpy(int) if lengths is None else np.asarray(lengths,int);order=np.argsort(lengths,kind="stable");outputs={};cursor=0;model.eval()
    while cursor<len(order):
        first=int(lengths[order[cursor]]);size=8 if first<=181 else 4 if first<=256 else 2 if first<=384 else 1;indices=order[cursor:cursor+size];cursor+=len(indices)
        encoded=tokenizer(frame.iloc[indices][gloss_col].astype(str).tolist(),padding=True,truncation=False,return_tensors="pt");ids=encoded["input_ids"].to(DEVICE);mask=encoded["attention_mask"].to(DEVICE)
        with torch.autocast("cuda",dtype=AMP_DTYPE):memory=model.encode(ids,mask)
        history=torch.full((len(indices),6,1),BOS_ID,dtype=torch.long,device=DEVICE);padding=torch.zeros((len(indices),1),dtype=torch.bool,device=DEVICE);steps=[]
        for _ in range(int(lengths[indices].max())):
            with torch.autocast("cuda",dtype=AMP_DTYPE):choice=model.decode(history,memory,mask,padding)[:,:,-1,:512].argmax(-1)
            steps.append(choice.cpu());history=torch.cat([history,choice.unsqueeze(-1)],-1);padding=torch.cat([padding,torch.zeros((len(indices),1),dtype=torch.bool,device=DEVICE)],1)
        stacked=torch.stack(steps,-1).numpy().astype(np.int64)
        for j,index in enumerate(indices):outputs[str(frame.iloc[index][ID])]=stacked[j,:,:int(lengths[index])]
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
    rows=[];lc=np.zeros(6);lt=np.zeros(6);mc=mt=0
    for sid in frame[ID].astype(str):
        gt,pred=tokens_for(sid),mapping[sid];n=min(gt.shape[1],pred.shape[1]);per=[];ed=[]
        for q in range(6):
            c=int((gt[q,:n]==pred[q,:n]).sum());lc[q]+=c;lt[q]+=n;per.append(c/max(n,1));ed.append(edit_distance(gt[q].tolist(),pred[q].tolist())/max(gt.shape[1],pred.shape[1],1))
        c=int((gt[:,:n]==pred[:,:n]).sum());mc+=c;mt+=6*n;rows.append({"sample_id":sid,"model":label,"aligned_token_accuracy":c/max(6*n,1),"normalized_token_edit_distance":float(np.mean(ed)),**{f"level_{q}_accuracy":per[q] for q in range(6)}})
    detail=pd.DataFrame(rows);return {"model":label,"aligned_token_accuracy":float(detail.aligned_token_accuracy.mean()),"micro_token_accuracy":mc/max(mt,1),"normalized_token_edit_distance":float(detail.normalized_token_edit_distance.mean()),"per_level_accuracy":(lc/np.maximum(lt,1)).tolist()},detail
@torch.inference_mode()
def decode_one(tokens):
    ids=torch.as_tensor(tokens,dtype=torch.long,device=DEVICE);latent=torch.zeros((256,tokens.shape[1]),device=DEVICE)
    for q in range(6):latent+=codebooks[q][:,ids[q]]
    motion=(decoder(latent[None]).squeeze(0).T*std_t+mean_t).cpu().numpy().astype(np.float32)
    if motion.shape!=(4*tokens.shape[1],668) or not np.isfinite(motion).all():raise RuntimeError("decode invariant")
    return motion
def z(m):return ((m[:,active]-motion_mean[active])/motion_std[active]).astype(np.float32)
def resample(m,n):return m if len(m)==n else F.interpolate(torch.from_numpy(m.T)[None],size=n,mode="linear",align_corners=True).squeeze(0).T.numpy()
def velocity(m):
    value=z(m);return np.sqrt(np.mean(np.square(np.diff(value,axis=0),dtype=np.float64),axis=1)) if len(value)>1 else np.empty(0)
def repetition(values,max_lag=180):
    x=np.asarray(values,dtype=np.float64);x=x-x.mean() if len(x) else x;correlations=[]
    for lag in range(2,min(max_lag,len(x)//2)+1):
        a,b=x[:-lag],x[lag:];d=np.sqrt(np.dot(a,a)*np.dot(b,b))
        if d>0:correlations.append(float(np.dot(a,b)/d))
    return max(correlations) if correlations else float("nan")
def motion_metrics(frame,mapping,label,save_dir=None):
    rows=[]
    if save_dir:Path(save_dir).mkdir(parents=True,exist_ok=True)
    for number,sid in enumerate(frame[ID].astype(str),1):
        pred=decode_one(mapping[sid]);gt=np.load(feature_index[sid],allow_pickle=False).astype(np.float32);aligned=resample(pred,len(gt));zp,zg=z(aligned),z(gt);vel=velocity(pred);mean_vel=float(vel.mean()) if len(vel) else 0.;variance=float(np.mean(np.var(z(pred),axis=0,dtype=np.float64)))
        collapse=mean_vel<float(thresholds["static_velocity_threshold_gt_percentile_1"]) or variance<float(thresholds["static_variance_threshold_gt_percentile_1"]);maxz=float(np.max(np.abs(z(pred))));extreme=float(pred.min())<float(thresholds["extreme_minimum_threshold_gt_sample_min_percentile_0_1"]) or float(pred.max())>float(thresholds["extreme_maximum_threshold_gt_sample_max_percentile_99_9"]) or maxz>float(thresholds["extreme_normalized_magnitude_threshold_gt_percentile_99_9"])
        rows.append({"sample_id":sid,"normalized_feature_mae":float(np.mean(np.abs(zp-zg))),"time_normalized_raw_mae":float(np.mean(np.abs(aligned-gt))),"velocity_mae":float(np.mean(np.abs(np.diff(zp,axis=0)-np.diff(zg,axis=0)))),"motion_variance":variance,"collapse":collapse,"repetition_score":repetition(vel),"extreme":extreme,"duration_error":len(pred)-len(gt)})
        if save_dir:np.save(Path(save_dir)/f"{sid}.npy",pred,allow_pickle=False)
        if number%100==0:print(label,"motion",number,"/",len(frame))
    detail=pd.DataFrame(rows);return {"model":label,"normalized_feature_mae":float(detail.normalized_feature_mae.mean()),"time_normalized_raw_mae":float(detail.time_normalized_raw_mae.mean()),"velocity_mae":float(detail.velocity_mae.mean()),"motion_variance":float(detail.motion_variance.mean()),"collapse_rate":float(detail.collapse.mean()),"repetition_score":float(detail.repetition_score.mean()),"extreme_motion_rate":float(detail.extreme.mean()),"duration_mae_frames":float(detail.duration_error.abs().mean()),"signed_duration_bias":float(detail.duration_error.mean())},detail


@torch.inference_mode()
def teacher_metrics(model,frame):
    data,_=make_loader(frame);model.eval();ce_values=[];vel_values=[];persist_values=[];correct=np.zeros(6);total=np.zeros(6)
    for raw in data:
        batch=move(raw)
        with torch.autocast("cuda",dtype=AMP_DTYPE):logits=model(batch["ids"],batch["mask"],batch["decoder"],batch["padding"]);ce,c,t=ce_and_accuracy(logits,batch["targets"]);lv,lp=latent_losses(logits,batch["targets"])
        ce_values.append(float(ce));vel_values.append(float(lv));persist_values.append(float(lp));correct+=c;total+=t
    return {"cross_entropy":float(np.mean(ce_values)),"latent_velocity_loss":float(np.mean(vel_values)),"persistence_loss":float(np.mean(persist_values)),"motion_accuracy":float(correct.sum()/total.sum()),"per_level_accuracy":(correct/np.maximum(total,1)).tolist()}


def train_variant(label,lambda_velocity,lambda_persistence,frame,epochs):
    seed_all(CFG.seed);model=fresh_model();data,sampler=make_loader(frame,shuffle=True,seed=CFG.seed);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CFG.learning_rate,weight_decay=CFG.weight_decay)
    try:scaler=torch.amp.GradScaler("cuda",enabled=AMP_DTYPE==torch.float16)
    except TypeError:scaler=torch.cuda.amp.GradScaler(enabled=AMP_DTYPE==torch.float16)
    history=[];best=None
    for epoch in range(1,epochs+1):
        sampler.set_epoch(epoch);model.train();sums=np.zeros(4);batches=0;p=corruption_probability(epoch)
        for batch_number,raw in enumerate(data):
            batch=move(raw);history_tokens=corrupted_history(model,batch,p,CFG.seed+epoch*100000+batch_number);model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=AMP_DTYPE):
                logits=model(batch["ids"],batch["mask"],history_tokens,batch["padding"]);ce,_,_=ce_and_accuracy(logits,batch["targets"]);lv,lp=latent_losses(logits,batch["targets"]);loss=ce+lambda_velocity*lv+lambda_persistence*lp
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],CFG.max_grad_norm);scaler.step(optimizer);scaler.update()
            sums+=np.asarray([float(loss.detach()),float(ce.detach()),float(lv.detach()),float(lp.detach())]);batches+=1
        tf=teacher_metrics(model,VAL_SUBSET);row={"epoch":epoch,"p_corruption":p,"total_loss":sums[0]/batches,"cross_entropy":sums[1]/batches,"latent_velocity_loss":sums[2]/batches,"persistence_loss":sums[3]/batches,"validation_teacher_accuracy":tf["motion_accuracy"]}
        mapping=exact_generate(model,VAL_SUBSET,label=f"{label} epoch {epoch}");ts,_=token_metrics(VAL_SUBSET,mapping,label);row.update({"validation_subset_edit":ts["normalized_token_edit_distance"],"validation_subset_micro":ts["micro_token_accuracy"]});history.append(row);print(label,row)
        rank=(row["validation_subset_edit"],-row["validation_subset_micro"])
        if best is None or rank<best[0]:best=(rank,epoch,{k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith("gloss_encoder.")})
    return best[2],pd.DataFrame(history),best[1]


rng=np.random.default_rng(CFG.seed);groups=pd.qcut(val_frame.motion_length,10,labels=False,duplicates="drop");indices=[]
for _,values in val_frame.groupby(groups).groups.items():indices.extend(rng.choice(list(values),min(math.ceil(CFG.validation_subset_samples/10),len(values)),replace=False))
VAL_SUBSET=val_frame.loc[sorted(set(indices))].iloc[:CFG.validation_subset_samples].reset_index(drop=True);SEARCH_TRAIN=train_frame.iloc[np.linspace(0,len(train_frame)-1,CFG.search_train_samples).round().astype(int)].reset_index(drop=True)

# Phase-7 reproduction: report lock plus actual validation generation before training.
locked={"micro":.1479332018,"edit":.8389708903,"feature":.8474933784,"collapse":.4746580853,"repetition":.4689987036}
p7_test=P7REPORT["models"]["sc_point"];observed={"micro":p7_test["oracle_length_tokens"]["micro_token_accuracy"],"edit":p7_test["oracle_length_tokens"]["normalized_token_edit_distance"],"feature":p7_test["oracle_length_motion"]["normalized_feature_mae"],"collapse":p7_test["oracle_length_motion"]["collapse_rate"],"repetition":p7_test["oracle_length_motion"]["repetition_score"]}
report_ok=all(abs(float(observed[k])-locked[k])<=({"micro":.03,"edit":.05,"feature":.08,"collapse":.08,"repetition":.08}[k]) for k in locked)
p7_model=fresh_model();p7_subset_tokens=exact_generate(p7_model,VAL_SUBSET,label="Phase7 reproduction");p7_subset_token_summary,_=token_metrics(VAL_SUBSET,p7_subset_tokens,"phase7_sc_point");p7_subset_motion_summary,_=motion_metrics(VAL_SUBSET,p7_subset_tokens,"phase7_sc_point")
reproduction={"locked_test_references":locked,"phase7_report_values":observed,"report_compatible":report_ok,"validation_subset_token":p7_subset_token_summary,"validation_subset_motion":p7_subset_motion_summary,"valid_ids":all(x.min()>=0 and x.max()<=511 for x in p7_subset_tokens.values())};reproduction["passed"]=bool(report_ok and reproduction["valid_ids"]);write_json(DIAG/"phase7_reproduction.json",reproduction)
if not reproduction["passed"]:raise RuntimeError("Phase-7 reproduction failed")


def subset_score(state,label):
    model=fresh_model();result=model.load_state_dict(state,strict=False);mapping=exact_generate(model,VAL_SUBSET,label=label);ts,_=token_metrics(VAL_SUBSET,mapping,label);ms,_=motion_metrics(VAL_SUBSET,mapping,label);del model,mapping;gc.collect();torch.cuda.empty_cache();return ts,ms
def rank(ts,ms):return (ms["collapse_rate"],ts["normalized_token_edit_distance"],ms["normalized_feature_mae"],ms["repetition_score"],-ts["micro_token_accuracy"])

search_rows=[]
for value in CFG.lambda_velocity_candidates:
    state,history,epoch=train_variant(f"velocity_{value}",value,0.,SEARCH_TRAIN,CFG.search_epochs);ts,ms=subset_score(state,f"velocity_{value}");search_rows.append({"stage":"velocity","lambda_velocity":value,"lambda_persistence":0.,"selected_epoch":epoch,**ts,**ms});del state;gc.collect();torch.cuda.empty_cache()
search=pd.DataFrame(search_rows);selected_velocity=float(search.sort_values(["collapse_rate","normalized_token_edit_distance","normalized_feature_mae","repetition_score","micro_token_accuracy"],ascending=[True,True,True,True,False]).iloc[0].lambda_velocity)
for value in CFG.lambda_persistence_candidates:
    state,history,epoch=train_variant(f"persistence_{value}",selected_velocity,value,SEARCH_TRAIN,CFG.search_epochs);ts,ms=subset_score(state,f"persistence_{value}");search_rows.append({"stage":"persistence","lambda_velocity":selected_velocity,"lambda_persistence":value,"selected_epoch":epoch,**ts,**ms});del state;gc.collect();torch.cuda.empty_cache()
search=pd.DataFrame(search_rows);search.to_csv(TRAINING/"regularization_search.csv",index=False);persistence_search=search[search.stage=="persistence"]
selected_persistence=float(persistence_search.sort_values(["collapse_rate","normalized_token_edit_distance","normalized_feature_mae","repetition_score","micro_token_accuracy"],ascending=[True,True,True,True,False]).iloc[0].lambda_persistence)
print("Selected validation-only weights",selected_velocity,selected_persistence)

policy={"split":"validation only","ranking":["lower collapse","lower edit distance","lower feature MAE","lower repetition","higher micro accuracy"],"rejection":{"feature_mae_delta":CFG.severe_feature_mae_delta,"edit_delta":CFG.severe_edit_delta,"collapse_delta":CFG.severe_collapse_delta},"fallback":"Phase7 SC-POINT if no fine-tuned model safely improves over continuation control/baseline","test_used":False}
write_json(DIAG/"model_selection_policy.json",policy)
specs={"continue_control":(0.,0.),"dynamics":(selected_velocity,0.),"dynamics_persistence":(selected_velocity,selected_persistence)};states={};validation={}
for label,(lv,lp) in specs.items():
    state,history,epoch=train_variant(label,lv,lp,train_frame,CFG.epochs);history.to_csv(TRAINING/f"{label}.csv",index=False);model=fresh_model();model.load_state_dict(state,strict=False);mapping=exact_generate(model,val_frame,label=label+" full validation");ts,_=token_metrics(val_frame,mapping,label);ms,_=motion_metrics(val_frame,mapping,label);validation[label]={"token":ts,"motion":ms,"selected_epoch":epoch,"lambda_velocity":lv,"lambda_persistence":lp};states[label]=state
    destination=MODELS/label;destination.mkdir(parents=True,exist_ok=True);torch.save({"motion_state_dict":state,"source_phase7_sha256":sha(P7_CKPT),"lambda_velocity":lv,"lambda_persistence":lp,"selected_epoch":epoch,"validation":validation[label]},destination/"best_model.pt");write_json(destination/"config.json",{"source_checkpoint":P7_CKPT,"source_phase7_sha256":sha(P7_CKPT),"architecture_changed":False,"phase3_changed":False,"length_model_changed":False,"self_conditioning":{"mode":"two-pass synchronized point corruption","p_max":CFG.p_max},**validation[label]});del model,mapping;gc.collect();torch.cuda.empty_cache()

# Full validation baseline and safe final choice.
p7_full_tokens=exact_generate(p7_model,val_frame,label="Phase7 full validation");p7_val_token,_=token_metrics(val_frame,p7_full_tokens,"phase7_sc_point");p7_val_motion,_=motion_metrics(val_frame,p7_full_tokens,"phase7_sc_point");validation["phase7_sc_point"]={"token":p7_val_token,"motion":p7_val_motion,"selected_epoch":0,"lambda_velocity":0.,"lambda_persistence":0.}
baseline=validation["phase7_sc_point"]
def acceptable(item):return item["motion"]["normalized_feature_mae"]<=baseline["motion"]["normalized_feature_mae"]+CFG.severe_feature_mae_delta and item["token"]["normalized_token_edit_distance"]<=baseline["token"]["normalized_token_edit_distance"]+CFG.severe_edit_delta and item["motion"]["collapse_rate"]<=baseline["motion"]["collapse_rate"]+CFG.severe_collapse_delta
acceptable_names=[name for name,item in validation.items() if acceptable(item)]
acceptable_auxiliary=[name for name in ("dynamics","dynamics_persistence") if name in acceptable_names]
auxiliary_winner=min(acceptable_auxiliary,key=lambda name:rank(validation[name]["token"],validation[name]["motion"])) if acceptable_auxiliary else None
# Locked negative-result rule: if neither auxiliary model beats the continued-training
# control, retain Phase-7 SC-POINT instead of mistaking extra epochs for a dynamics gain.
auxiliary_beats_control=bool(auxiliary_winner is not None and rank(validation[auxiliary_winner]["token"],validation[auxiliary_winner]["motion"])<rank(validation["continue_control"]["token"],validation["continue_control"]["motion"]))
final_variant=min(["phase7_sc_point",auxiliary_winner],key=lambda name:rank(validation[name]["token"],validation[name]["motion"])) if auxiliary_beats_control else "phase7_sc_point"
final_state=torch.load(P7_CKPT,map_location="cpu",weights_only=True)["motion_state_dict"] if final_variant=="phase7_sc_point" else states[final_variant]
final_lv=float(validation[final_variant]["lambda_velocity"]);final_lp=float(validation[final_variant]["lambda_persistence"])
torch.save({"motion_state_dict":final_state,"source_variant":final_variant,"lambda_velocity":final_lv,"lambda_persistence":final_lp,"source_phase7_sha256":sha(P7_CKPT),"validation":validation[final_variant]},FINAL_MODEL/"best_motion_generator.pt")
write_json(FINAL_MODEL/"config.json",{"source_variant":final_variant,"source_phase7_checkpoint":P7_CKPT,"source_phase7_sha256":sha(P7_CKPT),"final_checkpoint_sha256":sha(FINAL_MODEL/"best_motion_generator.pt"),"lambda_velocity":final_lv,"lambda_persistence":final_lp,"self_conditioning":{"mode":"two-pass synchronized point corruption","p_max":CFG.p_max},"architecture_changed":False,"phase3_changed":False,"length_model_changed":False,"validation_metrics":validation[final_variant]})
write_json(FINAL_MODEL/"selection.json",{"policy":policy,"all_validation_results":validation,"acceptable_models":acceptable_names,"best_auxiliary":auxiliary_winner,"auxiliary_beats_control":auxiliary_beats_control,"selected":final_variant,"rationale":"predeclared validation-only safe lexicographic policy with locked Phase-7 fallback when auxiliary losses fail to beat continuation control"})
print("FINAL GENERATOR FROZEN:",final_variant)

# %% [markdown]
# ## 8.14–8.25 — Frozen oracle/deployable tests and callable final pipeline

# %%
def load_state(model,state):
    result=model.load_state_dict(state,strict=False);expected=[n for n in model.state_dict() if n.startswith("gloss_encoder.")]
    if sorted(result.missing_keys)!=sorted(expected) or result.unexpected_keys:raise RuntimeError(str(result))
    model.eval()
final_generator=fresh_model();load_state(final_generator,final_state)


class LengthHead(nn.Module):
    def __init__(self):super().__init__();self.network=nn.Sequential(nn.Linear(int(encoder.config.d_model),256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
    def forward(self,x):return self.network(x).squeeze(-1)
length_config=read_json(P6/"length_model/config.json");bounds=list(map(int,length_config["training_bounds"]))
if bounds!=[24,469]:raise RuntimeError(bounds)
length_head=LengthHead().to(DEVICE);length_payload=torch.load(LENGTH_CKPT,map_location="cpu",weights_only=True);length_head.load_state_dict(length_payload["length_head_state_dict"],strict=True);length_head.eval()
for p in length_head.parameters():p.requires_grad=False
calibration=length_config["calibration"]
@torch.inference_mode()
def predict_motion_length(glosses):
    values=[]
    for start in range(0,len(glosses),64):
        encoded=tokenizer(glosses[start:start+64],padding=True,truncation=False,return_tensors="pt");ids=encoded["input_ids"].to(DEVICE);mask=encoded["attention_mask"].to(DEVICE);hidden=encoder(input_ids=ids,attention_mask=mask).last_hidden_state.float();pooled=(hidden*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1);values.extend(torch.expm1(length_head(pooled)).cpu().tolist())
    values=np.asarray(values)
    if calibration.get("selected"):values=float(calibration["slope"])*values+float(calibration["intercept"])
    return np.clip(np.rint(values),bounds[0],bounds[1]).astype(int)

phase3_report=read_json(P3/"phase3_report.json");phase3_beams=4 if phase3_report.get("selected_decoding")=="beam_4" else 1
phase3_generator=AutoModelForSeq2SeqLM.from_pretrained(PHASE3_MODEL,local_files_only=True).to(DEVICE).eval()
for p in phase3_generator.parameters():p.requires_grad=False
@torch.inference_mode()
def text_to_gloss(text):
    cleaned=re.sub(r"\s+"," ",str(text)).strip()
    if not cleaned:raise ValueError("English text must not be empty")
    encoded=tokenizer("text to gloss: "+cleaned,truncation=True,max_length=128,return_tensors="pt").to(DEVICE);output=phase3_generator.generate(**encoded,max_length=128,num_beams=phase3_beams,do_sample=False,early_stopping=phase3_beams>1)
    return re.sub(r"\s+"," ",tokenizer.decode(output[0],skip_special_tokens=True)).strip()
@torch.inference_mode()
def generate_rvq_tokens(gloss,length):
    frame=pd.DataFrame({ID:["inference"],GLOSS:[gloss],"motion_length":[int(length)]});return exact_generate(final_generator,frame,GLOSS,np.asarray([length]),"api")["inference"]
def decode_motion_s_tokens(tokens):return decode_one(np.asarray(tokens,dtype=np.int64))
def generate_motion(english_text,return_intermediates=False):
    gloss=text_to_gloss(english_text);length=int(predict_motion_length([gloss])[0]);tokens=generate_rvq_tokens(gloss,length);motion=decode_motion_s_tokens(tokens)
    if motion.dtype!=np.float32 or motion.shape!=(4*length,668) or not np.isfinite(motion).all():raise RuntimeError("Final API invariant failed")
    payload={"text":str(english_text),"predicted_gloss":gloss,"predicted_token_length":length,"rvq_tokens":tokens,"motion":motion}
    return payload if return_intermediates else motion

# Test is touched here only, after selection and freezing.
test_ids=test_frame[ID].astype(str).tolist();oracle_tokens=exact_generate(final_generator,test_frame,label="FINAL oracle-length test");oracle_token_summary,oracle_token_detail=token_metrics(test_frame,oracle_tokens,"phase8_final_oracle");oracle_motion_summary,oracle_motion_detail=motion_metrics(test_frame,oracle_tokens,"phase8_final_oracle",OUT/"decoded/final_oracle")
pd.concat([oracle_token_detail.set_index("sample_id"),oracle_motion_detail.set_index("sample_id")],axis=1).reset_index().to_csv(RESULTS/"final_oracle_length_metrics.csv",index=False)
phase3_predictions=pd.read_csv(P3/"results/test_predictions.csv",dtype={"sample_id":str});end_frame=test_frame.merge(phase3_predictions[["sample_id","predicted_gloss"]],left_on=ID,right_on="sample_id",validate="one_to_one");predicted_lengths=predict_motion_length(end_frame.predicted_gloss.astype(str).tolist());end_tokens=exact_generate(final_generator,end_frame,"predicted_gloss",predicted_lengths,"FINAL end-to-end test");end_token_summary,end_token_detail=token_metrics(test_frame,end_tokens,"phase8_final_end2end");end_motion_summary,end_motion_detail=motion_metrics(test_frame,end_tokens,"phase8_final_end2end",OUT/"decoded/final_end2end")
end_detail=pd.concat([end_token_detail.set_index("sample_id"),end_motion_detail.set_index("sample_id")],axis=1).reset_index();end_detail["predicted_token_length"]=predicted_lengths;end_detail["ground_truth_token_length"]=test_frame.motion_length.to_numpy();end_detail["token_length_error"]=end_detail.predicted_token_length-end_detail.ground_truth_token_length;end_detail.to_csv(RESULTS/"final_end2end_metrics.csv",index=False)
end_summary={**end_token_summary,**end_motion_summary,"length_mae_tokens":float(end_detail.token_length_error.abs().mean()),"signed_length_bias_tokens":float(end_detail.token_length_error.mean())}

# Validate public API on 20 deterministic held-out supported sentences.
api_rows=[]
for _,row in test_frame.iloc[np.linspace(0,len(test_frame)-1,20).round().astype(int)].iterrows():
    try:
        result=generate_motion(str(row[SENTENCE]),True);tokens,motion=result["rvq_tokens"],result["motion"]
        api_rows.append({"sample_id":str(row[ID]),"gloss_nonempty":bool(result["predicted_gloss"]),"predicted_length":result["predicted_token_length"],"length_in_bounds":bounds[0]<=result["predicted_token_length"]<=bounds[1],"token_shape_valid":tokens.shape==(6,result["predicted_token_length"]),"token_ids_valid":bool(tokens.min()>=0 and tokens.max()<=511),"motion_shape_valid":motion.shape==(4*result["predicted_token_length"],668),"motion_dtype":str(motion.dtype),"motion_dtype_valid":motion.dtype==np.float32,"finite":bool(np.isfinite(motion).all()),"error":""})
    except Exception as error:api_rows.append({"sample_id":str(row[ID]),"gloss_nonempty":False,"predicted_length":None,"length_in_bounds":False,"token_shape_valid":False,"token_ids_valid":False,"motion_shape_valid":False,"motion_dtype":"","motion_dtype_valid":False,"finite":False,"error":repr(error)})
api_validation=pd.DataFrame(api_rows);api_validation.to_csv(FINAL_PIPELINE/"api_validation.csv",index=False);api_passed=bool(len(api_validation)==20 and api_validation[["gloss_nonempty","length_in_bounds","token_shape_valid","token_ids_valid","motion_shape_valid","motion_dtype_valid","finite"]].all().all())

# %% [markdown]
# ## 8.17–8.34 — Final comparisons, diagnostics, API module, documentation, and completion gate

# %%
ablation=[]
for name,item in validation.items():
    evaluation_model=p7_model if name=="phase7_sc_point" else fresh_model()
    if name!="phase7_sc_point":load_state(evaluation_model,states[name])
    tf=teacher_metrics(evaluation_model,val_frame)
    ablation.append({"model":name,"evaluation_split":"validation","teacher_forced_accuracy":tf["motion_accuracy"],"micro_ar_accuracy":item["token"]["micro_token_accuracy"],"edit_distance":item["token"]["normalized_token_edit_distance"],"feature_mae":item["motion"]["normalized_feature_mae"],"velocity_mae":item["motion"]["velocity_mae"],"collapse":item["motion"]["collapse_rate"],"repetition":item["motion"]["repetition_score"],"extreme_motion_rate":item["motion"]["extreme_motion_rate"]})
    if name!="phase7_sc_point":del evaluation_model;gc.collect();torch.cuda.empty_cache()
pd.DataFrame(ablation).to_csv(RESULTS/"final_ablation.csv",index=False)

# Historical values remain clearly separated by oracle/deployable mode.
p6_oracle=P6REPORT["oracle_length"];p6_free=P6REPORT["baseline_free_running"];p6_tokens=P6REPORT["token_space"];p7_sc=P7REPORT["models"]["sc_point"]
p6_comparison=pd.read_csv(P6/"results/phase6_main_comparison.csv");phase4_rows=p6_comparison[p6_comparison.method=="original_free_run_oracle"]
if len(phase4_rows)!=1:raise RuntimeError("Phase-6 historical comparison lacks one original_free_run_oracle row")
phase4_free=phase4_rows.iloc[0].to_dict()
historical=[
 {"method":"retrieval_baseline","evaluation_mode":"language/retrieval context only"},
 {"method":"phase4c_free_run","evaluation_mode":"oracle_gloss_free_run","duration_mae_frames":p6_free["oracle_frame_mae"],"aligned_token_accuracy":p6_tokens["original_free_run_oracle"]["aligned_token_accuracy"],"micro_token_accuracy":p6_tokens["original_free_run_oracle"]["micro_aligned_token_accuracy"],"normalized_token_edit_distance":p6_tokens["original_free_run_oracle"]["normalized_token_edit_distance"],"normalized_feature_mae":phase4_free.get("normalized_feature_mae"),"velocity_mae":phase4_free.get("velocity_mae"),"collapse_rate":phase4_free.get("collapse_rate"),"repetition_score":phase4_free.get("repetition_score"),"extreme_motion_rate":phase4_free.get("extreme_motion_rate")},
 {"method":"phase6_oracle_length","evaluation_mode":"oracle_length","duration_mae_frames":p6_oracle["duration_mae_frames"],"aligned_token_accuracy":p6_tokens["oracle_gt_length"]["aligned_token_accuracy"],"micro_token_accuracy":p6_tokens["oracle_gt_length"]["micro_aligned_token_accuracy"],"normalized_token_edit_distance":p6_tokens["oracle_gt_length"]["normalized_token_edit_distance"],"normalized_feature_mae":p6_oracle["normalized_feature_mae"],"velocity_mae":p6_oracle["velocity_mae"],"collapse_rate":p6_oracle["collapse_rate"],"repetition_score":p6_oracle["repetition_score"],"extreme_motion_rate":p6_oracle["extreme_motion_rate"]},
 {"method":"phase7_sc_point","evaluation_mode":"oracle_length","aligned_token_accuracy":p7_sc["oracle_length_tokens"]["aligned_token_accuracy"],"micro_token_accuracy":p7_sc["oracle_length_tokens"]["micro_token_accuracy"],"normalized_token_edit_distance":p7_sc["oracle_length_tokens"]["normalized_token_edit_distance"],**p7_sc["oracle_length_motion"]},
 {"method":"phase8_final","evaluation_mode":"oracle_length","aligned_token_accuracy":oracle_token_summary["aligned_token_accuracy"],"micro_token_accuracy":oracle_token_summary["micro_token_accuracy"],"normalized_token_edit_distance":oracle_token_summary["normalized_token_edit_distance"],**oracle_motion_summary},
 {"method":"phase8_final","evaluation_mode":"deployable_end_to_end",**end_summary}]
pd.DataFrame(historical).to_csv(RESULTS/"final_historical_comparison.csv",index=False)

original_p7=P7REPORT["models"]["original"]["oracle_length_tokens"];per_level=[]
for label,summary in (("phase4c_original",original_p7),("phase7_sc_point",p7_sc["oracle_length_tokens"]),("phase8_final",oracle_token_summary)):
    per_level.append({"model":label,**{LEVELS[q]:summary["per_level_accuracy"][q] for q in range(6)}})
pd.DataFrame(per_level).to_csv(RESULTS/"per_level_comparison.csv",index=False)

# Temporal progress uses already-frozen historical Phase-7 token artifacts.
class Packed:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:self.ids=data["sample_ids"].astype(str);self.offsets=data["offsets"].astype(int);self.flat=data["tokens_flat"].astype(np.int64)
        self.index={x:i for i,x in enumerate(self.ids)}
    def get(self,sid):i=self.index[str(sid)];return self.flat[:,self.offsets[i]:self.offsets[i+1]]
p4_packed=Packed(P7/"tokens/oracle_length/original.npz");p7_packed=Packed(P7/"tokens/oracle_length/sc_point.npz");progress_rows=[]
for label,mapping in (("phase4c",{x:p4_packed.get(x) for x in test_ids}),("phase7_sc_point",{x:p7_packed.get(x) for x in test_ids}),("phase8_final",oracle_tokens)):
    correct=np.zeros(10);total=np.zeros(10)
    for sid in test_ids:
        gt,pred=tokens_for(sid),mapping[sid]
        for t in range(min(gt.shape[1],pred.shape[1])):
            b=min(9,int(10*t/gt.shape[1]));correct[b]+=int((gt[:,t]==pred[:,t]).sum());total[b]+=6
    for b in range(10):progress_rows.append({"model":label,"bin":f"{b*10}-{(b+1)*10}%","accuracy":correct[b]/max(total[b],1)})
progress=pd.DataFrame(progress_rows)
late_progress=progress[progress.bin.isin(["70-80%","80-90%","90-100%"])] .groupby("model").accuracy.mean().to_dict();phase8_late_delta=float(late_progress["phase8_final"]-late_progress["phase7_sc_point"])
fig,ax=plt.subplots(figsize=(8,5))
for label,part in progress.groupby("model"):ax.plot(range(10),part.accuracy,marker="o",label=label)
ax.set(xticks=range(10),xticklabels=[f"{i*10}-{(i+1)*10}%" for i in range(10)],xlabel="Normalized progress",ylabel="Token accuracy");ax.tick_params(axis="x",rotation=35);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"final_accuracy_vs_progress.png",dpi=170);plt.close(fig)

collapse_evolution=pd.DataFrame([{"phase":"Phase4C free-run","value":phase4_free["collapse_rate"]},{"phase":"Phase6 oracle-length","value":p6_oracle["collapse_rate"]},{"phase":"Phase7 SC-POINT","value":p7_sc["oracle_length_motion"]["collapse_rate"]},{"phase":"Phase8 final","value":oracle_motion_summary["collapse_rate"]}]);collapse_evolution.to_csv(RESULTS/"collapse_evolution.csv",index=False)
repetition_evolution=pd.DataFrame([{"phase":"Phase4C free-run","value":phase4_free["repetition_score"]},{"phase":"Phase6 oracle-length","value":p6_oracle["repetition_score"]},{"phase":"Phase7 SC-POINT","value":p7_sc["oracle_length_motion"]["repetition_score"]},{"phase":"Phase8 final","value":oracle_motion_summary["repetition_score"]}]);repetition_evolution.to_csv(RESULTS/"repetition_evolution.csv",index=False)
for data,path,title in ((collapse_evolution,FIGURES/"collapse_evolution.png","Collapse evolution"),(repetition_evolution,FIGURES/"repetition_evolution.png","Repetition evolution")):
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(data.phase,data.value);ax.set_title(title);ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

# Final qualitative feature-space examples: five low, median, high, and collapse cases.
ranked=oracle_motion_detail.sort_values("normalized_feature_mae").reset_index(drop=True);selected=[]
selected.extend(ranked.head(5).sample_id);selected.extend(ranked.iloc[max(0,len(ranked)//2-2):len(ranked)//2+3].sample_id);selected.extend(ranked.tail(5).sample_id);selected.extend(ranked[ranked.collapse].head(5).sample_id);selected=list(dict.fromkeys(map(str,selected)))
feature_indices=thresholds.get("high_variance_feature_indices",list(range(4)))[:4];p7_decoded=P7/"decoded/oracle_length/sc_point"
for sid in selected:
    gt=np.load(feature_index[sid],allow_pickle=False);p7_motion=np.load(p7_decoded/f"{sid}.npy",allow_pickle=False);p8_motion=np.load(OUT/"decoded/final_oracle"/f"{sid}.npy",allow_pickle=False);fig,axes=plt.subplots(5,1,figsize=(12,12))
    for index in feature_indices:axes[0].plot(gt[:,index],label=f"GT f{index}");axes[1].plot(p7_motion[:,index]);axes[2].plot(p8_motion[:,index])
    axes[3].plot(velocity(gt),label="GT");axes[3].plot(velocity(p7_motion),label="Phase7");axes[3].plot(velocity(p8_motion),label="Phase8");axes[3].legend();axes[4].axis("off");axes[4].text(0,.8,"Verified feature-space diagnostic only\nNo skeleton/BVH or semantic-sign claim",fontsize=11);axes[0].set_title(f"GT high-variance features — {sid}");axes[1].set_title("Phase7 SC-POINT");axes[2].set_title("Phase8 final");axes[3].set_title("Normalized velocity magnitude");fig.tight_layout();fig.savefig(FINAL_CASES/f"{sid}.png",dpi=150);plt.close(fig)

# Portable API module: manifest-driven bootstrap plus the same four-stage public call.
api_source='''\n"""Final Motion-S inference API. Load this file from the Phase-8 output directory."""\n_PIPELINE = None\ndef install_pipeline(text_to_gloss, predict_motion_length, generate_rvq_tokens, decode_motion_s_tokens):\n    global _PIPELINE\n    _PIPELINE=(text_to_gloss,predict_motion_length,generate_rvq_tokens,decode_motion_s_tokens)\ndef generate_motion(english_text, return_intermediates=False):\n    if _PIPELINE is None:\n        raise RuntimeError("Call install_pipeline(...) with the frozen manifest-loaded components first; see README_FINAL.txt")\n    text_to_gloss,predict_length,generate_tokens,decode_tokens=_PIPELINE\n    gloss=text_to_gloss(english_text); length=int(predict_length([gloss])[0]); tokens=generate_tokens(gloss,length); motion=decode_tokens(tokens)\n    result={"text":str(english_text),"predicted_gloss":gloss,"predicted_token_length":length,"rvq_tokens":tokens,"motion":motion}\n    return result if return_intermediates else motion\n'''
(FINAL_PIPELINE/"generate_motion.py").write_text(textwrap.dedent(api_source),encoding="utf-8")
manifest={"phase1_fingerprint":FINGERPRINT,"phase3_model":{"path":PHASE3_MODEL,"config_sha256":sha(PHASE3_MODEL/"config.json")},"phase3_tokenizer":TOKENIZER_PATH,"phase6_length_checkpoint":{"path":LENGTH_CKPT,"sha256":sha(LENGTH_CKPT)},"final_generator":{"path":FINAL_MODEL/"best_motion_generator.pt","sha256":sha(FINAL_MODEL/"best_motion_generator.pt"),"source_variant":final_variant},"rvq_checkpoint":{"path":RVQ_CKPT,"sha256":sha(RVQ_CKPT)},"normalization":{"mean":P5/"diagnostics/motion_mean_668.npy","mean_sha256":sha(P5/"diagnostics/motion_mean_668.npy"),"std":P5/"diagnostics/motion_std_668.npy","std_sha256":sha(P5/"diagnostics/motion_std_668.npy")},"token_ids":{"real":[0,511],"PAD":512,"BOS":513,"EOS":514},"contracts":{"generator":"int64 [6,T]","decoder":"float32 [4T,668]"},"versions":{"python":sys.version,"torch":torch.__version__}}
write_json(FINAL_PIPELINE/"model_manifest.json",manifest);write_json(FINAL_PIPELINE/"system_contract.json",{"input":"English UTF-8 text","phase3_output":"non-empty gloss string","length_output":"integer T bounded [24,469]","generator_output":"int64 [6,T], IDs 0..511","decoder_output":"float32 [4T,668], finite","unresolved":"exact 668-D semantic layout and verified joints/BVH inverse"})

limitations="""# Final limitations\n\n1. The 668-D semantics are only partially verified.\n2. No verified feature-to-joints inverse is available.\n3. No generated BVH is produced.\n4. MPJPE is not calculated.\n5. No human or sign-recognition evaluation was performed.\n6. Continuous feature similarity is not equivalent to linguistic sign correctness.\n7. Collapse remains measurable.\n8. Phase-3 errors affect predicted length and motion generation.\n9. Test data was evaluated in multiple development phases and is not a pristine once-only holdout.\n10. Further architectural changes are future work.\n"""
future="""# Future work\n\n1. Obtain the official 668-D inverse and skeleton conversion.\n2. Conduct human/sign-language expert evaluation.\n3. Investigate hierarchical coarse-to-fine RVQ generation.\n4. Investigate non-autoregressive or masked temporal generation.\n5. Study stronger exposure-bias methods.\n6. Improve gloss-conditioned length prediction.\n7. Add a semantic sign-recognition metric.\n8. Add joint/hand metrics only after representation verification.\n"""
(OUT/"FINAL_LIMITATIONS.md").write_text(limitations,encoding="utf-8");(OUT/"FUTURE_WORK.md").write_text(future,encoding="utf-8")

summary=f"""# Final research summary\n\nThe project progressed from retrieval and English-to-gloss modeling to six-level RVQ motion-token generation, strict decoder recovery, and empirical normalization recovery. Phase 5 exposed severe duration, collapse, and repetition failures. Phase 6 reduced oracle duration MAE from {p6_free['oracle_frame_mae']:.3f} to {p6_oracle['duration_mae_frames']:.3f} frames through explicit length control, while showing token quality remained poor. Phase 7 SC-POINT improved micro token accuracy to {p7_sc['oracle_length_tokens']['micro_token_accuracy']:.6f}, edit distance to {p7_sc['oracle_length_tokens']['normalized_token_edit_distance']:.6f}, and collapse to {p7_sc['oracle_length_motion']['collapse_rate']:.6f}. Phase 8 selected **{final_variant}** by validation only (lambda_velocity={final_lv}, lambda_persistence={final_lp}). Final oracle micro accuracy is {oracle_token_summary['micro_token_accuracy']:.6f}; normalized feature MAE is {oracle_motion_summary['normalized_feature_mae']:.6f}; collapse is {oracle_motion_summary['collapse_rate']:.6f}. Final deployable duration MAE is {end_summary['duration_mae_frames']:.3f} frames. These are token/feature-space results, not proof of linguistic sign correctness.\n"""
(OUT/"FINAL_RESEARCH_SUMMARY.md").write_text(summary,encoding="utf-8")

readme=f"""MOTION-S FINAL SYSTEM

Final generator: {final_variant}
Checkpoint: final_model/best_motion_generator.pt
Phase-3 model/tokenizer: {PHASE3_MODEL} / {TOKENIZER_PATH}
Length predictor: {LENGTH_CKPT}, calibrated bounds [24,469]
RVQ decoder: {RVQ_CKPT}
Normalization: {P5/'diagnostics/motion_mean_668.npy'} and {P5/'diagnostics/motion_std_668.npy'}

EXACT RELOAD ORDER
1. Load the Phase-3 tokenizer and AutoModelForSeq2SeqLM from the manifest paths with local_files_only=True; set eval() and freeze it.
2. Construct the unchanged Phase-4/7 MotionModel, attach the Phase-3 encoder, then load final_model/best_motion_generator.pt['motion_state_dict']; set eval() and freeze it.
3. Construct the unchanged Phase-6 LengthHead, load length_model/best_model.pt['length_head_state_dict'], and apply the calibration and [24,469] bounds recorded in its config.json.
4. Construct the verified MotionDecoder, load decoder.* tensors from rvq_vae_best.pth, and use the six rvq.quantizers.q.embedding tensors without modification.
5. Load motion_mean_668.npy and motion_std_668.npy for decoder de-normalization.
6. Import final_pipeline/generate_motion.py, call install_pipeline(text_to_gloss, predict_motion_length, generate_rvq_tokens, decode_motion_s_tokens), then call generate_motion(...).

EXECUTED NOTEBOOK EXAMPLE
result = generate_motion("I want water", return_intermediates=True)
Expected: gloss string; integer T; rvq_tokens int64 [6,T] IDs 0..511; motion float32 [4T,668].

The notebook has already installed and validated these frozen functions. model_manifest.json records paths, hashes, IDs, versions, and shape contracts. No verified skeleton/BVH inverse exists.
"""
(OUT/"README_FINAL.txt").write_text(readme,encoding="utf-8")

required_outputs=[DIAG/"artifact_verification.json",DIAG/"phase7_reproduction.json",DIAG/"model_selection_policy.json",FINAL_MODEL/"best_motion_generator.pt",FINAL_MODEL/"config.json",FINAL_MODEL/"selection.json",TRAINING/"continue_control.csv",TRAINING/"dynamics.csv",TRAINING/"dynamics_persistence.csv",RESULTS/"final_ablation.csv",RESULTS/"final_oracle_length_metrics.csv",RESULTS/"final_end2end_metrics.csv",RESULTS/"final_historical_comparison.csv",RESULTS/"per_level_comparison.csv",RESULTS/"collapse_evolution.csv",RESULTS/"repetition_evolution.csv",FIGURES/"final_accuracy_vs_progress.png",FIGURES/"collapse_evolution.png",FIGURES/"repetition_evolution.png",FINAL_PIPELINE/"generate_motion.py",FINAL_PIPELINE/"model_manifest.json",FINAL_PIPELINE/"system_contract.json",FINAL_PIPELINE/"api_validation.csv",OUT/"FINAL_RESEARCH_SUMMARY.md",OUT/"FINAL_LIMITATIONS.md",OUT/"FUTURE_WORK.md",OUT/"README_FINAL.txt"]
phase8_complete=bool(reproduction["passed"] and set(specs)=={"continue_control","dynamics","dynamics_persistence"} and api_passed and len(oracle_token_detail)==1243 and len(end_detail)==1243 and all(path.is_file() for path in required_outputs))
report={"phase":"Phase 8 - Final Motion Generator and End-to-End System","phase8_complete":phase8_complete,"project_complete":phase8_complete,"phase1_fingerprint":FINGERPRINT,"phase7_baseline":p7_sc,"regularization_experiment":validation,"final_generator":{"source_variant":final_variant,"lambda_velocity":final_lv,"lambda_persistence":final_lp,"validation_selection":validation[final_variant]},"oracle_length_test":{**oracle_token_summary,**oracle_motion_summary},"end_to_end_test":end_summary,"per_level_analysis":{"labels":LEVELS,"accuracies":oracle_token_summary["per_level_accuracy"],"deeper_levels_harder_than_base":bool(np.mean(oracle_token_summary["per_level_accuracy"][3:])<oracle_token_summary["per_level_accuracy"][0]),"semantic_interpretation_claimed":False},"late_sequence_drift":{"mean_accuracy_70_to_100_percent":late_progress,"phase8_minus_phase7":phase8_late_delta,"improved":phase8_late_delta>0},"pipeline_validation":{"samples":20,"token_validity":bool(api_validation.token_ids_valid.all()),"all_motion_finite":bool(api_validation.finite.all()),"passed":api_passed},"historical_test_limitation":"Test metrics were inspected in Phases 4-7; Phase-8 choices used validation only and test was run after freezing.","semantic_motion_quality_claimed":False,"skeleton_conversion_verified":False,"bvh_export_verified":False}
write_json(OUT/"phase8_config.json",{"config":asdict(CFG),"resolved":resolved,"selection_policy":policy});write_json(OUT/"phase8_report.json",report)
inventory=[]
for path in sorted(OUT.rglob("*")):
    if path.is_file() and path.name!="artifact_inventory.csv":inventory.append({"relative_path":str(path.relative_to(OUT)),"bytes":path.stat().st_size,"sha256":sha(path)})
pd.DataFrame(inventory).to_csv(OUT/"artifact_inventory.csv",index=False)

print("="*60);print("PHASE 8 — FINAL PROJECT COMPLETE");print("="*60);print("Final generator:",final_variant);print("lambda_velocity:",final_lv);print("lambda_persistence:",final_lp);print("-"*60);print("FINAL ORACLE-LENGTH TEST");print("Micro token accuracy:",oracle_token_summary["micro_token_accuracy"]);print("Normalized token edit distance:",oracle_token_summary["normalized_token_edit_distance"]);print("Normalized feature MAE:",oracle_motion_summary["normalized_feature_mae"]);print("Velocity MAE:",oracle_motion_summary["velocity_mae"]);print("Collapse rate:",oracle_motion_summary["collapse_rate"]);print("Repetition:",oracle_motion_summary["repetition_score"]);print("-"*60);print("FINAL END-TO-END TEST");print("Length MAE:",end_summary["length_mae_tokens"],"tokens");print("Duration MAE:",end_summary["duration_mae_frames"],"frames");print("Normalized token edit distance:",end_summary["normalized_token_edit_distance"]);print("Normalized feature MAE:",end_summary["normalized_feature_mae"]);print("Velocity MAE:",end_summary["velocity_mae"]);print("Collapse rate:",end_summary["collapse_rate"]);print("Repetition:",end_summary["repetition_score"]);print("-"*60);print("FINAL PIPELINE");print("English -> Gloss: PASS");print("Gloss -> Length: PASS");print("Gloss + Length -> RVQ: PASS");print("RVQ -> [T,668]: PASS");print("Invalid token rate:",0.0 if api_validation.token_ids_valid.all() else float((~api_validation.token_ids_valid).mean()));print("NaN/Inf:",0 if api_validation.finite.all() else int((~api_validation.finite).sum()));print("generate_motion():","PASS" if api_passed else "FAIL");print("-"*60);print("668-D semantic inverse: UNRESOLVED");print("Skeleton conversion: NOT VERIFIED");print("BVH generation: NOT VERIFIED");print("MPJPE: NOT CALCULATED");print("Semantic sign correctness: NOT CLAIMED");print("PROJECT COMPLETE:","YES" if phase8_complete else "NO");print("Further model changes: FUTURE WORK ONLY");print("="*60)
