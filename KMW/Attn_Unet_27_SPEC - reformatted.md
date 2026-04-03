# U-Net Mapper Structure (Locked for v1.1.1)

This section defines the **authoritative mapper backbone** for the KMW method.

### Design intent

This U-Net is a **shallow conditional mapper** that operates in the **latent reordered frame**.

It must:

- take the reordered circuit matrix `A*` as the **only spatial input**
- take reordered hardware tokens `T_hw*` as the **only conditioning source**
- output latent-frame mapping logits `S*`
- leave native-frame decode and final assignment **outside** the backbone

This is the converted version of the old 5-channel U-Net design.

## Table of Contents

- [1. Locked architectural choices](#1-locked-architectural-choices)
- [2. Input / output contract](#2-input--output-contract)
  - [Spatial input](#spatial-input)
  - [Conditioning input](#conditioning-input)
  - [Backbone output](#backbone-output)
  - [Interpretation](#interpretation)
- [3. Channel schedule](#3-channel-schedule)
- [4. Spatial resolution schedule](#4-spatial-resolution-schedule)
- [5. High-level block sequence](#5-high-level-block-sequence)
- [6. Exact forward structure](#6-exact-forward-structure)
  - [Stage 1 — down1](#stage-1--down1)
  - [Stage 2 — down2](#stage-2--down2)
  - [Stage 3 — bottleneck](#stage-3--bottleneck)
  - [Stage 4 — up1](#stage-4--up1)
  - [Stage 5 — up2 / head](#stage-5--up2--head)
- [7. ASCII diagram](#7-ascii-diagram)
- [8. Cross-attention block](#8-cross-attention-block)
  - [Attention dimensions](#attention-dimensions)
  - [Why 4 heads are locked](#why-4-heads-are-locked)
  - [Cross-attention rule](#cross-attention-rule)
- [9. Skip-connection policy](#9-skip-connection-policy)
- [10. What is intentionally NOT included](#10-what-is-intentionally-not-included)
- [11. Decode and assignment are outside the backbone](#11-decode-and-assignment-are-outside-the-backbone)
- [12. Reimplementation notes](#12-reimplementation-notes)
  - [Must preserve](#must-preserve)
  - [Allowed implementation freedom](#allowed-implementation-freedom)
  - [Not allowed without a later ablation revision](#not-allowed-without-a-later-ablation-revision)
- [Short update summary](#short-update-summary)

---

## 1. Locked architectural choices

The following choices are fixed:

- keep the **same shallow backbone**
- keep the **same three attention injection points**
- keep the **same additive skip**
- change the spatial input from **5-channel `X`** to **1-channel `A*`**
- remove logical tokens completely
- use **hardware-only tokens** from `B*`, `c2*`, `c1*`
- keep decode/native-frame assignment outside the backbone
- implement **actual multi-head attention**
- use **4 attention heads**
- make interpolation mode explicit
- do **not** deepen the network yet
- do **not** add full U-Net copy/crop skips yet

---

## 2. Input / output contract

### Spatial input

```text
A* : (B, 1, 27, 27)

B = batch size

channel count is always 1

A* is the reordered circuit interaction matrix
```

### Conditioning input

```text
T_hw* : (B, 27, 128)

where each hardware token comes from:

x_hw*[j] = concat(B*[j, :], c2*[j, :], c1*[j])    # shape (55,)
t_hw*[j] = MLP_hw(x_hw*[j])                        # shape (128,)

So:

T_hw* = stack(t_hw*[0], ..., t_hw*[26])           # shape (27, 128)
```

### Backbone output

```text
S* : (B, 1, 27, 27)
```

### Interpretation

```text
S*[u, j] = latent-frame logit for mapping latent logical slot u
           to latent hardware slot j
```

The backbone must return (B, 1, 27, 27) directly.
Do not squeeze the channel dimension inside the mapper.

---

## 3. Channel schedule

Use this fixed channel schedule:

```text
input     : 1
down1     : 64
down2     : 128
bottleneck: 256
up1       : 128
head      : 1
```

This is intentionally shallow.

---

## 4. Spatial resolution schedule

Use this fixed spatial schedule:

```text
27 x 27  ->  14 x 14  ->  7 x 7  ->  14 x 14  ->  27 x 27
```

All resizing must use explicit interpolation settings:

```text
mode = bilinear
align_corners = False
```

Use this same interpolation rule for every spatial resize inside the backbone.

---

## 5. High-level block sequence

The mapper has 5 main spatial stages:

- down1
- down2 + attention
- bottleneck + attention
- up1 + additive skip + attention
- up2 + output head

There is only one active skip connection:

```text
up1_pre_attn = up1_pre_attn + down2_out
```

There is no skip from down1.

There are no copy-and-crop U-Net skips.

---

## 6. Exact forward structure

### Stage 1 — down1

**Input:**

```text
A*                           : (B, 1, 27, 27)
```

**Operation:**

```text
d1 = ReLU( Conv3x3(1 -> 64)(A*) )
```

**Output:**

```text
d1                           : (B, 64, 27, 27)
```

### Stage 2 — down2

**Resize:**

```text
d1_ds = Interpolate(d1, size=(14, 14))
```

**Operation:**

```text
d2_pre = ReLU( Conv3x3(64 -> 128)(d1_ds) )
d2     = CrossAttn(d2_pre, T_hw*)
```

**Output:**

```text
d2                           : (B, 128, 14, 14)
```

### Stage 3 — bottleneck

**Resize:**

```text
d2_ds = Interpolate(d2, size=(7, 7))
```

**Operation:**

```text
b_pre = ReLU( Conv3x3(128 -> 256)(d2_ds) )
b     = CrossAttn(b_pre, T_hw*)
```

**Output:**

```text
b                            : (B, 256, 7, 7)
```

### Stage 4 — up1

**Resize:**

```text
b_us = Interpolate(b, size=(14, 14))
```

**Operation:**

```text
u1_pre  = ReLU( Conv3x3(256 -> 128)(b_us) )
u1_skip = u1_pre + d2
u1      = CrossAttn(u1_skip, T_hw*)
```

**Output:**

```text
u1                           : (B, 128, 14, 14)
```

### Stage 5 — up2 / head

**Resize:**

```text
u1_us = Interpolate(u1, size=(27, 27))
```

**Operation:**

```text
S* = Conv3x3(128 -> 1)(u1_us)
```

**Output:**

```text
S*                           : (B, 1, 27, 27)
```

No activation is applied after the final output conv.

---

## 7. ASCII diagram

```text
A* : (B,1,27,27)
  │
  ├─ Conv3x3(1→64) + ReLU
  ▼
d1 : (B,64,27,27)
  │
  ├─ Interpolate → (14,14)
  ├─ Conv3x3(64→128) + ReLU
  ├─ CrossAttn(T_hw*)   [4 heads]
  ▼
d2 : (B,128,14,14)
  │
  ├─ Interpolate → (7,7)
  ├─ Conv3x3(128→256) + ReLU
  ├─ CrossAttn(T_hw*)   [4 heads]
  ▼
 b : (B,256,7,7)
  │
  ├─ Interpolate → (14,14)
  ├─ Conv3x3(256→128) + ReLU
  ├─ Add skip from d2
  ├─ CrossAttn(T_hw*)   [4 heads]
  ▼
u1 : (B,128,14,14)
  │
  ├─ Interpolate → (27,27)
  ├─ Conv3x3(128→1)
  ▼
S* : (B,1,27,27)
```

---

## 8. Cross-attention block

Cross-attention is injected at exactly 3 points:

- after down2
- after bottleneck
- after up1 skip fusion

### Attention dimensions

Use:

```text
token_dim      = 128
attn_dim       = 128
num_heads      = 4
head_dim       = 32
```

since:

```text
128 / 4 = 32
```

Use the same head count at all three injection points.

### Why 4 heads are locked

4 heads are the chosen compromise because they provide enough specialization without making each head too narrow.

- 2 heads are too coarse
- 8 heads are too thin for this model size
- 4 heads with 32 dims/head is the best balanced default

### Cross-attention rule

For a feature map F : (B, C, H, W):

```text
F_flat = reshape(F) to (B, H*W, C)

Q = Linear(C   -> 128)(F_flat)
K = Linear(128 -> 128)(T_hw*)
V = Linear(128 -> 128)(T_hw*)
```

Then split into 4 heads:

```text
Q -> (B, heads=4, H*W, 32)
K -> (B, heads=4, 27,  32)
V -> (B, heads=4, 27,  32)
```

Per head:

```text
Attn_h = softmax( Q_h K_h^T / sqrt(32) ) V_h
```

Concatenate all heads:

```text
AttnOut = concat(Attn_1, ..., Attn_4)   # (B, H*W, 128)
```

Project back to the feature-map channel dimension:

```text
DeltaF_flat = Linear(128 -> C)(AttnOut)
DeltaF      = reshape(DeltaF_flat) back to (B, C, H, W)
```

Residual gated update:

```text
F_out = F + alpha_attn * DeltaF
```

where:

```text
alpha_attn
```

is a learnable scalar per attention block, initialized to:

```text
0.0
```

This makes attention start as a no-op and become useful only if training learns to use it.

---

## 9. Skip-connection policy

Use exactly this skip policy:

```text
u1_skip = u1_pre + d2
```

Rules:

- additive skip only
- only one skip
- skip happens at the 14 x 14 level
- no concatenation skip
- no down1 -> up2 skip
- no full classical U-Net skip ladder

---

## 10. What is intentionally NOT included

The following are intentionally excluded in v1.1.1:

- no 5-channel spatial input
- no logical tokens
- no token type embeddings
- no extra down/up levels
- no copy-and-crop skip ladder
- no attention at every block
- no decode inside the backbone
- no final Hungarian/Sinkhorn inside the backbone
- no D inside the hardware token used by the mapper

D stays in loss / routing proxy logic, not in mapper token conditioning.

---

## 11. Decode and assignment are outside the backbone

The backbone stops at latent-frame logits:

```text
S* : (B, 1, 27, 27)
```

Then outside the backbone:

```text
S_nat = R_L^T S* R_H
P_map = LogSinkhorn(S_nat / tau_m)          # training
M_map = Hungarian(S_nat)                    # inference
```

So the U-Net is a latent-frame score generator, not the full mapping pipeline.

---

## 12. Reimplementation notes

### Must preserve

- exact stage count
- exact channel schedule
- exact attention placement
- exact single skip structure
- exact interpolation policy
- exact 4-head attention choice

### Allowed implementation freedom

- helper function names
- internal class organization
- whether each stage is split into helper methods
- whether attention is implemented in a separate class or inline inside model.py

### Not allowed without a later ablation revision

- adding more blocks
- adding more skips
- changing channel widths
- changing number of attention heads
- changing attention insertion points
- changing the spatial input back to multi-channel form

---

## Short update summary

- 4 attention heads are now locked
- the converted U-Net stays shallow
- input is `A*` only
- conditioning is `T_hw*` only
- one additive skip only
- three attention blocks only
- ready to drop directly into `Design Plan v1.1.1.md`
