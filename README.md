# Architecture of Motion-S RVQ-VAE

This implementation is a **Residual Vector Quantized Variational Autoencoder (RVQ-VAE)** designed for motion sequence compression and tokenization. Let me break down each component:

---

## 1. **Overall Architecture Flow**

```
Input Motion (B, D_POSE, N) 
    ↓
[Encoder] → Continuous Latent (B, d, n)
    ↓
[RVQ] → Quantized Latent (B, d, n) + Discrete Tokens
    ↓
[Decoder] → Reconstructed Motion (B, D_POSE, N)
```

Where:
- **B** = Batch size
- **D_POSE** = Motion feature dimension (e.g., 263 for body pose)
- **N** = Original sequence length (frames)
- **d** = Latent dimension (default 256)
- **n** = Downsampled sequence length (N // downsampling_ratio)

---

## 2. **MotionEncoder: Convolutional Downsampling**

### Purpose
Compresses motion sequences both **spatially** (D_POSE → d) and **temporally** (N → n).

### Architecture
```python
Input: (B, D_POSE, N)  # Treats D_POSE as channels, N as sequence length

Layer Structure (4 layers default):
  Conv1D(D_POSE → 512, kernel=3, stride=2, padding=1)  # Temporal downsampling
  ReLU + BatchNorm
  
  Conv1D(512 → 512, kernel=3, stride=2, padding=1)      # More downsampling
  ReLU + BatchNorm
  
  Conv1D(512 → 512, kernel=3, stride=1, padding=1)      # Maintain resolution
  ReLU + BatchNorm
  
  Conv1D(512 → 256, kernel=3, stride=1, padding=1)      # Project to latent_dim
  ReLU + BatchNorm

Output: (B, 256, n)  # n ≈ N/4 for downsampling_ratio=4
```

### Key Design Choices
- **Stride=2 for first log₂(ratio) layers**: Achieves 4x downsampling with two stride-2 convolutions
- **BatchNorm**: Stabilizes training by normalizing activations
- **1D Convolutions**: Efficient for sequential data vs 2D/RNNs

---

## 3. **ResidualVectorQuantizer (RVQ): Hierarchical Quantization**

### Purpose
Converts continuous latents into **discrete tokens** using a codebook hierarchy.

### Core Concept: Residual Quantization
Instead of quantizing once, RVQ quantizes the **residual error** iteratively:

```
Step 0: Quantize input          → b⁰ = Q₀(r⁰), where r⁰ = continuous latent
Step 1: Quantize residual       → b¹ = Q₁(r¹), where r¹ = r⁰ - b⁰
Step 2: Quantize new residual   → b² = Q₂(r²), where r² = r¹ - b¹
...
Step V: Final residual          → bⱽ = Qᵥ(rⱽ)

Final Output: Σ(b⁰, b¹, ..., bⱽ)  # Sum of all quantized codes
```

### Architecture
```python
num_quantizers = 6  # V+1 layers (0 to 5)

For each layer v:
  1. Calculate distances to codebook:
     distances = ||z - embedding||²  # (B*n, num_embeddings)
  
  2. Find nearest code:
     indices = argmin(distances)     # (B*n,)
  
  3. Lookup quantized vector:
     quantized = embedding[:, indices]  # (B, d, n)
  
  4. Compute next residual:
     residual = residual - quantized
```

### VectorQuantizer: Single-Layer Quantization

Each layer has:
- **Codebook**: `embedding` tensor of shape `(d, num_embeddings=512)`
  - 512 learnable code vectors, each of dimension 256
  
- **EMA Updates** (Exponential Moving Average):
  ```python
  cluster_size = (1-decay) * new_counts + decay * old_counts
  embedding_avg = (1-decay) * new_codes + decay * old_codes
  embedding = embedding_avg / cluster_size  # Normalize
  ```
  - Prevents codebook collapse (dead codes)
  - No explicit gradient descent on codebook

- **Straight-Through Estimator**:
  ```python
  quantized_st = inputs + (quantized - inputs).detach()
  ```
  - Forward: Use quantized values
  - Backward: Gradients flow through inputs (bypassing non-differentiable argmin)

- **Commitment Loss**:
  ```python
  loss = λ * ||quantized - inputs||²
  ```
  - Encourages encoder to produce latents close to codebook entries

---

## 4. **MotionDecoder: Convolutional Upsampling**

### Purpose
Reconstructs original motion from quantized latent.

### Architecture
```python
Input: (B, 256, n)

Layer Structure (mirror of encoder):
  ConvTranspose1D(256 → 512, kernel=3, stride=2, padding=1, output_padding=1)
  ReLU + BatchNorm
  
  ConvTranspose1D(512 → 512, kernel=3, stride=2, padding=1, output_padding=1)
  ReLU + BatchNorm
  
  Conv1D(512 → 512, kernel=3, stride=1, padding=1)
  ReLU + BatchNorm
  
  Conv1D(512 → D_POSE, kernel=3, stride=1, padding=1)  # Final layer, no activation

Output: (B, D_POSE, N)  # Restored to original dimensions
```

### Key Design Choices
- **ConvTranspose1D**: Learns upsampling (better than fixed interpolation)
- **output_padding**: Ensures exact size matching after strided convolutions
- **No activation on final layer**: Allows unrestricted output range

---

## 5. **Loss Function: Multi-Component Objective**

```python
Total Loss = L_rec + λ_global·L_global + λ_commit·L_commit + λ_vel·L_vel
```

### Components

1. **Reconstruction Loss** (L_rec):
   ```python
   L_rec = SmoothL1(reconstructed, target)
   ```
   - Main objective: Match overall motion

2. **Global/Root Loss** (L_global):
   ```python
   L_global = SmoothL1(reconstructed[:, :4], target[:, :4])
   ```
   - Focuses on first 4 dimensions:
     - Root rotation velocity
     - Root linear velocity (X/Z)
     - Root height
   - Weighted 1.5x to prioritize global motion

3. **Velocity Loss** (L_vel):
   ```python
   pred_vel = reconstructed[:, :, 1:] - reconstructed[:, :, :-1]
   target_vel = target[:, :, 1:] - target[:, :, :-1]
   L_vel = SmoothL1(pred_vel, target_vel)
   ```
   - Ensures temporal smoothness
   - Prevents jittery motion
   - Weighted 2.0x for importance

4. **Commitment Loss** (L_commit):
   ```python
   L_commit = Σ(||quantized_v - inputs_v||²) for all RVQ layers
   ```
   - From RVQ: encourages encoder outputs near codebook
   - Weighted 0.02x (small to avoid over-constraining)

---

## 6. **Training Features**

### Quantization Dropout
```python
if training and rand() < 0.2:
    num_active_layers = randint(1, num_quantizers+1)
```
- Randomly uses 1 to V+1 quantization layers
- Improves robustness and generalization
- Forces lower layers to capture more information

### Masking Support
```python
loss = mean_flat(error * mask) / (mask.sum() + ε)
```
- Handles variable-length sequences with padding
- Only computes loss on valid frames

---

## 7. **Token Representation**

### Encoding to Tokens
```python
tokens = [indices_0, indices_1, ..., indices_V]  # List of (B, n) tensors
```
- Each token sequence represents one RVQ layer
- Token values ∈ [0, 511] (for 512 codebook entries)
- Total vocabulary size: 512^(V+1) combinations

### Decoding from Tokens
```python
quantized = Σ(embedding[:, tokens_v]) for v in layers
reconstructed = decoder(quantized)
```
- Lookup codes from each layer's codebook
- Sum all codes to get final latent
- Pass through decoder

---

## 8. **Key Hyperparameters**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `input_dim` | 263 | Motion feature dimension |
| `latent_dim` | 256 | Bottleneck dimension |
| `downsampling_ratio` | 4 | Temporal compression (N → N/4) |
| `num_quantizers` | 6 | RVQ hierarchy depth (V+1) |
| `num_embeddings` | 512 | Codebook size per layer |
| `commitment_cost` | 1.0 | Weight for commitment loss |
| `decay` | 0.99 | EMA decay for codebook updates |
| `quantization_dropout` | 0.2 | Probability of layer dropout |

---



---

## 9. **Usage Example**

```python
# Training
model = RVQVAE(input_dim=263, output_dim=263)
reconstructed, tokens, commit_loss = model(motion)
total_loss, loss_dict = compute_rvq_loss(reconstructed, motion, commit_loss)

# Inference: Motion → Tokens
tokens = model.encode_to_tokens(motion)  # List of (B, n) discrete tokens

# Inference: Tokens → Motion
reconstructed = model.decode_from_tokens(tokens)
```



---
license: apache-2.0
---
