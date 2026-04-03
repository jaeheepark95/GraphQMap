# Design Plan v1.1.1 — Clarification Patch

## 0. Document Status

This document is a clarification revision of **Design Plan v1.1**.

**v1.1.1 introduces no semantic design changes.**  
It only resolves wording ambiguities and implementation-loose sections in v1.1 so that different implementers produce the same system.

If any section below conflicts with the wording of v1.1, the **v1.1.1 wording takes precedence**, but the intended model/design remains the same as v1.1.

---

## 1. Clarification: Reference U-Net Backbone

### Replace Section 10.5 with:

### 10.5 U-Net backbone

The mapper must preserve the **same macro-architecture and channel schedule** as the current reference U-Net used for this project.

This means:

- same number of down blocks,
- same number of up blocks,
- same bottleneck structure,
- same per-stage channel widths,
- same normalization type,
- same residual/non-residual block structure,
- same final logits head shape.

The phrase “reuse the current U-Net macro-architecture in the codebase” means **copy the exact architecture behavior**, not “implement a similar U-Net.”

Required guarantees remain:

- spatial input size: `27 x 27`
- spatial output size: `27 x 27`
- final logits shape: `(Bsz, 1, 27, 27)`

Interpretation:

`S*[u, j] = score for mapping latent logical slot u to latent hardware slot j`

### Implementation rule

When the KMW implementation is created under its own folder structure, the implementer must do one of the following:

1. directly reuse the reference U-Net implementation, or
2. reimplement it exactly and record the source file + commit/hash used as the reference.

This is to prevent silent divergence caused by “roughly similar” U-Net implementations.

---

## 2. Clarification: Raw vs Normalized Tensor Naming

### Replace Section 6.7 with:

### 6.7 Backend extractor invariants

Two sets of invariants must be checked.

#### 6.7.1 Raw tensor invariants

Immediately after backend extraction, assert:

- `B_raw.shape   == (27, 27)`
- `c1_raw.shape  == (27,)`
- `c2_raw.shape  == (27, 27)`
- `D_raw.shape   == (27, 27)`

and:

- `B_raw` is symmetric binary
- `c2_raw` is symmetric finite
- `D_raw` is symmetric finite
- `diag(B_raw) == 0`
- `diag(c2_raw) == 0`
- `diag(D_raw) == 0`

#### 6.7.2 Normalized tensor invariants

After normalization, assert:

- `B.shape   == (27, 27)`
- `c1.shape  == (27,)`
- `c2.shape  == (27, 27)`
- `D.shape   == (27, 27)`

and:

- `B` is symmetric binary
- `c1` is finite
- `c2` is symmetric finite
- `D` is symmetric finite
- `diag(B) == 0`
- `diag(c2) == 0`
- `diag(D) == 0`

This split is mandatory so that raw extraction bugs and normalization bugs are not conflated.

---

## 3. Clarification: Hardware Branch Feature Definitions

### Replace Section 9.8 with:

### 9.8 Hardware branch input features

For each hardware qubit `j`, define:

- `deg(j) = sum_k B[j, k]`
- `Nbr(j) = { k | B[j, k] = 1 }`

Then define:

- `mean_c2(j) = mean_k c2[j, k] over k in Nbr(j)`, else `0`
- `min_c2(j)  = min_k  c2[j, k] over k in Nbr(j)`, else `0`
- `mean_D(j)  = mean_k D[j, k] over all k in {0, ..., 26}`

Then:

`feat_H[j] = [ c1[j], deg(j), mean_c2(j), min_c2(j), mean_D(j) ]`

This wording makes it explicit that `c2` is indexed as a row-wise pairwise matrix, not a one-dimensional vector.

---

## 4. Clarification: Cross-Attention Variable Naming

### Replace Section 10.7 with:

### 10.7 Cross-attention rule

At each injection point, with feature map `F` of shape `(Bsz, C, H, W)`:

- `F_flat = reshape(F)` to `(Bsz, H*W, C)`
- `Q = F_flat @ W_Q`
- `K = T_hw* @ W_K`
- `V = T_hw* @ W_V`

Define:

`AttnOut = softmax(Q K^T / sqrt(d_a)) V`

Then:

- `DeltaF_flat = AttnOut @ W_O`
- `DeltaF = reshape(DeltaF_flat)` back to `(Bsz, C, H, W)`

Residual update:

`F <- F + alpha_attn * DeltaF`

Use:

- multi-head attention,
- `alpha_attn` as a learnable scalar per block,
- initialize `alpha_attn = 0`.

This is only a notation clarification; it does not change the attention mechanism already specified in v1.1. The v1.1 attention rule already defined Q/K/V and the residual gated update. :contentReference[oaicite:3]{index=3}

---

## 5. Clarification: Mapper Sinkhorn Implementation

### Replace Section 11.2 / 11.5 wording with:

### 11.2 Training assignment

Use **log-domain Sinkhorn** on decoded logits:

`P_map = LogSinkhorn(S_nat / tau_m)`

### 11.5 Mapper Sinkhorn default

The mapper assignment head must use the **same stabilized log-domain Sinkhorn implementation family** as the reindexer unless a pre-existing stabilized implementation is already in the codebase and is documented as numerically equivalent.

Default fallback values:

- `tau_m = 0.10`
- `T_m = 20`

### Implementation rule

Do **not** use naive exp-domain Sinkhorn for the mapper if log-domain Sinkhorn is already used for the reindexer.

This is a clarification, not a design change: v1.1 already explicitly required log-domain Sinkhorn for the reindexer and used Sinkhorn for the mapper assignment head, but did not state clearly enough that the mapper should also use the stabilized variant. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## 6. Clarification: Meaning of “PST” Loss Terms

### Replace the opening text of Section 12 with:

### 12. Task Loss

The loss terms named `L_PST_1Q`, `L_PST_2Q`, and `L_PST_total` are **normalized proxy costs correlated with PST degradation**.

They are **not literal physical PST probabilities**, because they are built from normalized cost tensors `c1` and `c2`, including z-score normalized quantities.

Therefore:

- these terms may be negative,
- these terms are optimization proxies,
- these terms should be interpreted as “PST-related cost proxies,” not direct PST values.

### Implementation note

For code readability, it is recommended to use names such as:

- `L_pst_proxy_1q`
- `L_pst_proxy_2q`
- `L_pst_proxy_total`

while preserving the same formulas and the same role in the total objective.

This clarification is necessary because v1.1 defines these terms using normalized `c1` and `c2` tensors. :contentReference[oaicite:6]{index=6}

---

## 7. Clarification: Pass-B Mapper Freeze Semantics

### Replace the relevant part of Section 14.5 with:

### 14.5 Pass B — reindexer update

1. recompute `R_L, R_H` without detach
2. rebuild reordered tensors
3. rebuild reordered hardware tokens
4. run the mapper as a **frozen differentiable operator**
5. decode `S_nat`
6. compute `P_map`
7. compute `L_task`
8. compute `L_loc`, `L_cons`
9. compute `L_reindex`
10. backward only into reindexer parameters
11. clip reindexer gradients
12. optimizer step for reindexer only

### Frozen differentiable operator rule

“Run mapper with mapper parameters frozen” means:

- mapper parameters must **not** be updated in Pass B,
- gradients must still be allowed to flow **through mapper outputs back into reindexer outputs**,
- therefore **do not** wrap the mapper forward in `torch.no_grad()`.

Operationally:

- set mapper parameters to `requires_grad = False` for Pass B, or otherwise exclude them from gradient accumulation/update,
- keep the mapper forward differentiable with respect to its inputs,
- after Pass B, restore normal mapper parameter state for the next Pass A.

### Determinism rule

To avoid stochastic mismatch during reindexer optimization:

- the mapper should run in **evaluation mode** during Pass B,
- so dropout and running-stat updates do not introduce noise into the reindexer objective,
- then return the mapper to training mode afterward.

This does not change the two-pass design; it only makes the intended behavior explicit.

The underlying two-pass structure is already part of v1.1. :contentReference[oaicite:7]{index=7}

---

## 8. Clarification: Consistency-Loss Sampling

### Replace Section 13.2 opening with:

### 13.2 Consistency loss

Per Pass-B update, sample **one** random logical relabeling `Pi_L` and **one** random hardware relabeling `Pi_H`.

Use those same sampled relabelings consistently across all tensors in that Pass-B computation.

So for one Pass-B step:

- sample one `Pi_L`,
- sample one `Pi_H`,
- construct `A', m', B', c1', c2', D'`,
- rerun the reindexer,
- compute `L_cons` from the reordered outputs.

### Default rule

- one relabeling pair per Pass-B step,
- resampled every Pass-B step,
- not cached across epochs by default.

This removes ambiguity without changing the consistency-loss definition already present in v1.1. :contentReference[oaicite:8]{index=8}

---

## 9. Clarification: Dataset Split Authority

### Append to Section 5.7:

### 5.7.1 Split authority

For any concrete experiment run, the authoritative split definition is the manifest files stored under the KMW project data-manifest directory.

Once manifests are generated for a run family, they must be committed or archived and reused for reproducibility.

### Default rule

- use a fixed random seed for manifest generation,
- do not regenerate manifests silently between runs,
- validation must be held out at the circuit-file level,
- no identical circuit file may appear across train/val/test.

This does not change the dataset policy in v1.1; it only makes reproducibility operational.

---

## 10. Clarification: Path Wording for KMW Implementation

### Replace path language like “Use the existing mapper file, e.g. ...” with:

If the KMW project is implemented under its own directory tree, the **behavioral contract** is authoritative, not the original file path.

So references to older paths such as:

- `mqm/networks/unet_mapper.py`
- `mqm/networks/reindexer.py`
- `training/train_label_free_v11.py`

should be interpreted as **reference implementation locations**, not mandatory final file paths.

For the KMW project, equivalent modules may live under the KMW package structure, as long as behavior matches the specification exactly.

This clarification is necessary because v1.1 contains both design-authority wording and legacy/example path wording. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

---

## 11. Summary of v1.1.1

v1.1.1 does **not** change:

- architecture,
- token definition,
- reindexer math,
- native-frame decode,
- dataset policy,
- two-pass optimization,
- or loss formulas.

It only clarifies:

- exact U-Net reuse meaning,
- raw vs normalized tensor checks,
- row-wise `c2` feature definitions,
- attention variable naming,
- mapper Sinkhorn implementation choice,
- PST-term interpretation,
- Pass-B mapper-freeze semantics,
- consistency-loss sampling,
- and split/path authority.