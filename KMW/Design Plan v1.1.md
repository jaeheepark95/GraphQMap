Design Plan v1.1 — Implementation Specification

0. Document Status

This document is the implementation authority for the revised project.

If this document conflicts with:

the older Design Plan v1_1.pdf, or

the older canonical-indexing version of the model,

this document wins.

This document is written for implementation, not for high-level presentation.

1. Objective

Build a label-free learned initial qubit mapper for a fixed 27-qubit IBM BackendV2 target.

The system must:

accept a quantum circuit with K <= 27 logical qubits,

extract a fixed-size circuit representation,

extract fixed native hardware tensors from the backend,

learn a soft internal reordering of logical and hardware indices,

run a U-Net-style mapper in the reordered latent frame,

decode logits back to the native hardware frame,

produce a one-to-one initial mapping using Sinkhorn during training and Hungarian during inference,

train against a proxy objective correlated with:

PST,

SWAP overhead,

depth increase.

The one-to-one assignment requirement and the Sinkhorn/Hungarian contract come from the prior mapper design; the native-frame decode and two-pass optimization come from the reindexer design.

2. Non-Goals

This revision does not do the following:

full routing / explicit SWAP insertion,

diffusion or timestep conditioning,

backend-specific supervised imitation of GraphQMap labels,

multi-backend training in a single run,

direction-aware hardware tokenization beyond the symmetrized cost extraction defined below.

No diffusion/timestep embedding is used in the prior mapper design, and that remains true here.

3. Locked Design Decisions
3.1 Canonical indexing is removed

Do not implement or use a canonical indexer.
Do not compute any permutation p.
Do not convert canonical indices back to native IDs.

Use only:

native logical tensors,

native hardware tensors,

learned soft reindexing,

native-frame decode.

This follows the reindexer replacement design.

3.2 Mapper conditioning style

Use Imagen-style circuit-to-hardware conditioning:

the circuit tensor is the main U-Net spatial input,

hardware is injected as cross-attention conditioning,

queries come from U-Net feature maps,

keys and values come from hardware-only tokens.

Do not use logical tokens in the mapper.

3.3 U-Net input

The U-Net input is only the circuit matrix A.

Do not use the old 5-channel X = [W, A, c2, C1_col, Mmask] input contract from the previous design as the mapper input.
That older grid remains historically relevant, but this revision replaces it with a single-channel circuit image + hardware attention tokens. The old plan explicitly used a fixed 5-channel grid and logical+physical token concatenation; this spec does not.

3.4 Hardware token definition

For each physical qubit j, define the raw hardware token as:

x_hw[j] = concat(B[j, :], c2[j, :], c1[j])

with shape:

B[j, :]   -> (27,)
c2[j, :]  -> (27,)
c1[j]     -> (1,)
x_hw[j]   -> (55,)

This is the exact mapper hardware-token contract for this revision.

3.5 Distance matrix role

Keep D in:

loss computation,

routing proxies,

reindexer hardware branch features,

but do not include D[j, :] in the mapper hardware token.

3.6 No infinities anywhere

Never use inf, -inf, or large artificial sentinels in:

B,

c1,

c2,

D,

attention inputs,

loss computation,

routing proxy computation.

Finite-only distance handling is explicitly required in the reindexer plan.

3.7 Batch size

Use:

batch_size = 1

as the default and recommended setting. The reindexer plan lists batch size 1 as the recommended default.

3.8 Auxiliary losses

Implement reindexer auxiliary losses in code, but for the initial stable run set:

alpha_loc = 0.0
beta_cons = 0.0

Then turn them on only in later runs.

The reindexer plan defines L_reindex = L_task + alpha * L_loc + beta * L_cons, but also warns not to start with large coefficients.

4. Notation

Use the following notation consistently in code and docs.

4.1 Dimensions
n = 27     # fixed physical qubit count
K <= 27    # logical qubit count for a given circuit
Bsz = 1    # batch size
4.2 Circuit tensors
A      : (27, 27)  # circuit interaction matrix
m      : (27,)     # logical-valid mask
4.3 Hardware tensors
B      : (27, 27)  # binary hardware adjacency / connectivity
c1     : (27,)     # per-physical-qubit cost
c2     : (27, 27)  # per-physical-edge cost
D      : (27, 27)  # shortest-path distance on B
4.4 Reindexer outputs
R_L    : (27, 27)  # logical soft permutation
R_H    : (27, 27)  # hardware soft permutation

Orientation convention:

rows    = latent slots
columns = original IDs

So:

R_L[t, u] = probability that original logical node u goes to latent slot t
R_H[t, j] = probability that original hardware node j goes to latent slot t

This orientation is explicitly defined in the reindexer plan.

4.5 Reordered tensors
A*   = R_L A R_L^T        # circuit matrix in latent logical frame
m*   = R_L m
B*   = R_H B R_H^T
c1*  = R_H c1
c2*  = R_H c2 R_H^T
D*   = R_H D R_H^T
4.6 Mapper outputs
S*      : (27, 27)  # latent-frame mapping logits
S_nat   : (27, 27)  # decoded native-frame logits
P_map   : (27, 27)  # soft assignment in training
M_map   : (27, 27)  # hard assignment in inference
4.7 Decode rule
S_nat = R_L^T S* R_H

and in inference, replace R_L, R_H with hard permutations R_L_hat, R_H_hat. This is the exact reconstruction rule in the reindexer plan.

5. Dataset Protocol

The MQT generator script you uploaded creates circuits across many algorithm types and qubit counts from 2 to 127, with multiple variants for VQE and QAOA. That is why hard filtering to K <= 27 is mandatory for this project.

5.1 Phase A — stability smoke test

Train briefly on:

MQT only

Purpose:

verify preprocessing,

verify finite-value handling,

verify no exploding loss,

verify the two-pass training loop works.

5.2 Phase B — main comparison training

Train on:

QUEKO + MLQD + MQT

Use source-balanced sampling.

5.3 Final evaluation

Do not train on:

QASMBench
RevLib

Use them only for final transfer/generalization evaluation.

5.4 Hard inclusion rules

A circuit is included iff all are true:

2 <= K <= 27
contains at least one 2Q gate
parses successfully
hardware preprocessing succeeds
circuit interaction matrix is not all-zero off-diagonal
5.5 Keep disconnected logical graphs

Do not remove circuits just because the logical interaction graph is disconnected.
Only remove degenerate cases with no useful 2Q structure.

5.6 Labels

Do not use GraphQMap labels as direct training targets.
This method is label-free and uses proxy-loss training only.

5.7 Train/val splitting

Use source-wise splitting:

smoke run:

MQT train / val split only

main run:

QUEKO train / val

MLQD train / val

MQT train / val

final test:

QASMBench

RevLib

No identical circuit file may appear across train/val/test.

5.8 Source balancing

In the main run, sample sources uniformly:

p(source = QUEKO) = 1/3
p(source = MLQD)  = 1/3
p(source = MQT)   = 1/3

Then sample a circuit uniformly from the chosen source.

Do not just concatenate the datasets and shuffle.

6. Backend Extraction

This module converts an IBM BackendV2 target into the native hardware tensors.

6.1 Input
backend: IBM BackendV2
6.2 Output
B_raw   : (27, 27)
c1_raw  : (27,)
c2_raw  : (27, 27)
D_raw   : (27, 27)
6.3 Hardware adjacency B

Construct a symmetric binary adjacency:

B[i, j] = 1 if at least one usable 2Q direction exists between i and j
B[i, j] = 0 otherwise
B[i, i] = 0

Then enforce symmetry:

B = max(B, B^T)

Use B only as topology/validity.
Do not encode hardware quality into B.

6.4 Per-qubit cost c1

Define:

e_ro[j] = readout error for qubit j if available, else 0.0

e_1q[j] = mean error over supported 1Q basis gates on qubit j; if unavailable, 0.0

Then:

c1_raw[j] = 0.5 * e_ro[j] + 0.5 * e_1q[j]

This yields a scalar per physical qubit.

6.5 Per-edge cost c2

For each unordered pair (i, j):

collect valid directional 2Q gate errors:

e_ij if (i -> j) exists,

e_ji if (j -> i) exists.

Then define:

if no valid direction exists:
    c2_raw[i, j] = 0
elif only one valid direction exists:
    c2_raw[i, j] = that direction's error
else:
    c2_raw[i, j] = min(e_ij, e_ji)

Then enforce symmetry:

c2_raw[j, i] = c2_raw[i, j]
c2_raw[i, i] = 0

This intentionally treats c2 as a symmetric edge badness for initial mapping.

6.6 Distance matrix D

Build the shortest-path distance matrix on the graph defined by B.

Use:

Floyd-Warshall,

or BFS from every node.

Define:

D_raw[i, i] = 0
D_raw[i, j] = shortest hop count if reachable
D_raw[i, j] = 28 if unreachable

because n + 1 = 28.

6.7 Backend extractor invariants

After extraction, assert:

B.shape   == (27, 27)
c1.shape  == (27,)
c2.shape  == (27, 27)
D.shape   == (27, 27)

B is symmetric binary
c2 is symmetric finite
D is symmetric finite
diag(B) == 0
diag(c2) == 0
diag(D) == 0
7. Circuit Featurization

This module converts a circuit into (A, m).

7.1 Logical qubit count

Let:

K = number of logical qubits in the circuit

Reject the circuit if:

K > 27
7.2 Logical mask m

Define:

m[u] = 1 for u < K
m[u] = 0 for u >= K
7.3 Off-diagonal entries of A

For u != v, define:

A_raw[u, v] = log(1 + sum_{g in G_2Q(u,v)} w_2q(type(g)))

where:

G_2Q(u,v) = all 2Q gates involving logical qubits u and v

w_2q(type) = gate-type weight

7.4 Default 2Q gate-type weights

For the first stable revision:

w_2q(type) = 1.0 for all 2Q gate types

So off-diagonals are effectively log-scaled 2Q interaction counts.

Do not differentiate 2Q gate types in v1.1 unless you intentionally run an ablation.

7.5 Diagonal entries of A

For logical qubit u, define:

N1Q(u)       = number of 1Q gates on u
N2Q_part(u)  = number of 2Q gates in which u participates

Then:

A_raw[u, u] = log(1 + alpha_diag * N1Q(u) + beta_diag * N2Q_part(u))

Use:

alpha_diag = 0.25
beta_diag  = 1.00

These defaults intentionally give more weight to 2Q participation than to 1Q load.

7.6 Padding

Initialize:

A_raw = zeros(27, 27)

Fill only the active K x K logical block.
All padded rows/cols remain zero.

7.7 Symmetry

After constructing off-diagonal entries, enforce:

A_raw = 0.5 * (A_raw + A_raw.T)
7.8 Degenerate circuit rejection

Reject the circuit if:

sum_{u != v} A_raw[u, v] == 0

This removes no-2Q or effectively zero-interaction circuits.

8. Normalization and Finite-Value Policy
8.1 Global epsilon

Use:

eps = 1e-8

for normalization denominators.

8.2 Circuit matrix A

Normalize by max value:

A = A_raw / (max(A_raw) + eps)

If max(A_raw) == 0, leave A as zeros.

8.3 Hardware adjacency B

Do not normalize.

Keep as:

B in {0, 1}
8.4 Per-qubit cost c1

Normalize with z-score across the 27 qubits:

mu_c1    = mean(c1_raw)
sigma_c1 = std(c1_raw)
c1       = (c1_raw - mu_c1) / (sigma_c1 + eps)
8.5 Per-edge cost c2

Important: compute statistics over valid edges only.

Define:

E_valid = {(i, j) | B[i, j] == 1}

Compute:

mu_c2    = mean(c2_raw[i, j] for (i, j) in E_valid)
sigma_c2 = std(c2_raw[i, j] for (i, j) in E_valid)

Then:

if B[i, j] == 1:
    c2[i, j] = (c2_raw[i, j] - mu_c2) / (sigma_c2 + eps)
else:
    c2[i, j] = 0

This preserves the non-edge rule:

c2[i, j] = 0 when B[i, j] = 0

and avoids having non-edge zeros corrupt the normalization statistics.

8.6 Distance matrix D

Normalize by max finite entry:

D = D_raw / (max(D_raw) + eps)
8.7 Summary of finite rules

Use exactly:

B[i, j] in {0, 1}
c2[i, j] = 0 when B[i, j] = 0
D[i, j]  = 28 when unreachable before normalization
no inf anywhere
9. Reindexer

The reindexer is a separate module that learns soft latent reorderings of:

logical indices,

hardware indices.

It does not solve the mapping itself. This role is explicitly stated in the reindexer design.

9.1 File

Create:

mqm/networks/reindexer.py
9.2 Components

Implement:

LogSinkhorn
LogicalReindexBranch
HardwareReindexBranch
SoftPermutationReindexer

This file structure follows the reindexer plan.

9.3 Reindexer hidden dimension

Use:

d_r = 128

The reindexer plan recommends d_r = 128.

9.4 Logical branch input features

For each logical slot u, define:

feat_L[u] = [
    log(1 + sum_v A[u, v]),
    top1_offdiag(A[u, :]),
    top2_offdiag(A[u, :]),
    m[u]
]

Rules:

top1_offdiag and top2_offdiag ignore the diagonal.

if fewer than 2 nonzero off-diagonal values exist, use 0.

9.5 Logical branch MLP

Use:

LayerNorm(4)
Linear(4, d_r)
ReLU
Dropout(0.1)
Linear(d_r, d_r)
LayerNorm(d_r)
9.6 Logical slot prototypes

Learn:

E_L : (27, d_r)
9.7 Logical score matrix

Compute:

H_L = LogicalMLP(feat_L)                 # (27, d_r)
G_L = H_L @ E_L.T                        # (27, 27)
G_L_tilde[t, u] = G_L[u, t]              # transpose orientation
R_L = LogSinkhorn(G_L_tilde / tau_r)
9.8 Hardware branch input features

For each hardware qubit j, define:

deg(j)        = sum_k B[j, k]
mean_c2(j)    = mean(c2[k] over valid neighbors), else 0
min_c2(j)     = min(c2[k] over valid neighbors), else 0
mean_D(j)     = mean(D[j, :])

Then:

feat_H[j] = [
    c1[j],
    deg(j),
    mean_c2(j),
    min_c2(j),
    mean_D(j)
]
9.9 Hardware branch MLP

Use:

LayerNorm(5)
Linear(5, d_r)
ReLU
Dropout(0.1)
Linear(d_r, d_r)
LayerNorm(d_r)
9.10 Hardware slot prototypes

Learn:

E_H : (27, d_r)
9.11 Hardware score matrix

Compute:

H_H = HardwareMLP(feat_H)               # (27, d_r)
G_H = H_H @ E_H.T                       # (27, 27)
G_H_tilde[t, j] = G_H[j, t]
R_H = LogSinkhorn(G_H_tilde / tau_r)
9.12 Reindexer Sinkhorn

Use log-domain Sinkhorn, not naive exp-Sinkhorn.

Defaults from the reindexer plan:

tau_r schedule: 1.0 -> 0.15
T_r = 20 iterations

9.13 Reordered tensors

Given (A, m, B, c1, c2, D) and (R_L, R_H):

A*   = R_L @ A  @ R_L.T
m*   = R_L @ m
B*   = R_H @ B  @ R_H.T
c1*  = R_H @ c1
c2*  = R_H @ c2 @ R_H.T
D*   = R_H @ D  @ R_H.T

Note:

during training, these reordered matrices become soft/dense,

this is expected and acceptable in the differentiable relaxation. The reindexer plan explicitly notes that reordered tensors become soft/dense during training.

10. Mapper
10.1 File

Use the existing mapper file, e.g.:

mqm/networks/unet_mapper.py

Modify it rather than redesigning the entire backbone.

10.2 Input contract

The mapper must accept:

A*        : (Bsz, 1, 27, 27)   # U-Net spatial input
T_hw*     : (Bsz, 27, d_tok)   # hardware tokens
10.3 Hardware token encoder

For reordered hardware tensors, build hardware tokens in latent hardware order:

x_hw*[j] = concat(B*[j, :], c2*[j, :], c1*[j])

with shape (55,).

Then embed with:

LayerNorm(55)
Linear(55, d_tok)
ReLU
Dropout(0.1)
Linear(d_tok, d_tok)
LayerNorm(d_tok)

Use:

d_tok = 128
10.4 No logical tokens

Do not build logical tokens for the mapper.

10.5 U-Net backbone

Reuse the current U-Net macro-architecture in the codebase.
Do not redesign the down/up channel schedule in this revision.

Required guarantees:

spatial input size: 27 x 27

spatial output size: 27 x 27

final logits shape: (Bsz, 1, 27, 27)

Interpretation:

S*[u, j] = score for mapping latent logical slot u to latent hardware slot j
10.6 Cross-attention placement

Inject cross-attention at exactly these 3 locations:

last down block,

bottleneck,

first up block.

This preserves the old Tier-2 cross-attention placement rule.

10.7 Cross-attention rule

At each injection point, with feature map F of shape (Bsz, C, H, W):

F_flat = reshape(F) to (Bsz, H*W, C)

Q = F_flat @ W_Q
K = T_hw* @ W_K
V = T_hw* @ W_V

Attn(Q, K, V) = softmax(Q K^T / sqrt(d_a)) V
DeltaF = reshape(AttnOut @ W_O) back to (Bsz, C, H, W)

F <- F + alpha_attn * DeltaF

Use:

multi-head attention,

alpha_attn as a learnable scalar per block,

initialize alpha_attn = 0.

The prior mapper design explicitly used residual cross-attention with a learned gate initialized to zero.

10.8 No timestep embedding

Do not add timestep embedding.

11. Decode and Assignment
11.1 Decode to native frame

After the mapper produces S*, decode:

S_nat = R_L.T @ S* @ R_H
11.2 Training assignment

Use Sinkhorn on decoded logits:

P_map = Sinkhorn(S_nat / tau_m)
11.3 Inference assignment

Use Hungarian on decoded logits:

M_map = Hungarian(S_nat)

This is exactly the native-frame assignment pattern specified in the reindexer plan.

11.4 Mapping output

Return final mapping as:

logical_u -> native_physical_j

for u = 0 .. K-1.

There is no canonical-to-native conversion table anymore.

11.5 Mapper Sinkhorn default

If your current code already has a stabilized mapper Sinkhorn setting, keep it.

Otherwise use:

tau_m = 0.10
T_m   = 20

as the default fallback.

12. Task Loss

The older mapper design used:

L = lambda_P * L_PST + lambda_S * L_swap + lambda_D * L_depth

and the reindexer plan keeps the task loss semantically unchanged, but computes it in the native frame after decode.

12.1 Active logical set

Define:

U_active = {u | m[u] = 1}
12.2 Off-diagonal circuit mass

Define:

A_off[u, v] = A[u, v] for u != v
A_off[u, u] = 0
mass_2q = max(sum_{u,v} A_off[u,v], 1e-6)
mass_1q = max(sum_u m[u], 1e-6)
12.3 PST 1Q term

Define:

L_PST_1Q_num =
    sum_u m[u] * sum_j P_map[u, j] * c1[j]

L_PST_1Q =
    L_PST_1Q_num / mass_1q
12.4 PST 2Q term

Define the usable edge-cost matrix:

C2_use = B * c2

Then:

L_PST_2Q_num =
    sum_{u,v} A_off[u,v] *
    sum_{i,j} P_map[u,i] * P_map[v,j] * C2_use[i,j]

L_PST_2Q =
    L_PST_2Q_num / mass_2q
12.5 PST total

Define:

L_PST_total = L_PST_1Q + L_PST_2Q
12.6 SWAP proxy

Define the expected physical distance for a logical pair:

E_D(u,v) =
    sum_{i,j} P_map[u,i] * P_map[v,j] * D[i,j]

Then:

L_swap_num =
    sum_{u,v} A_off[u,v] * E_D(u,v)

L_swap =
    L_swap_num / mass_2q
12.7 Depth proxy

Define:

L_depth = kappa_depth * L_swap

Use:

kappa_depth = 1.0

for v1.1.

12.8 Total task loss

Use:

lambda_P = 1.0
lambda_S = 1.0
lambda_D = 0.25

Then:

L_task = lambda_P * L_PST_total
       + lambda_S * L_swap
       + lambda_D * L_depth
12.9 Dummy logical rows

Dummy rows automatically contribute zero because:

m[u] = 0,

padded circuit rows/cols are zero.

This matches the prior mapper design.

13. Reindexer Auxiliary Losses
13.1 Locality loss

Compute on reordered tensors.

Logical locality
W_mass = max(sum_{u,v} A*[u,v], 1e-6)

L_loc_log =
    (1 / W_mass) *
    sum_{u,v} A*[u,v] * ((u - v) / 26)^2
Hardware locality
B_mass = max(sum_{i,j} B*[i,j], 1e-6)

L_loc_hw =
    (1 / B_mass) *
    sum_{i,j} B*[i,j] * ((i - j) / 26)^2
Total locality
L_loc = 0.5 * (L_loc_log + L_loc_hw)

This follows the locality structure defined in the reindexer plan.

13.2 Consistency loss

Sample random relabelings:

Pi_L : logical permutation matrix
Pi_H : hardware permutation matrix

Construct relabeled inputs:

A'   = Pi_L A Pi_L^T
m'   = Pi_L m
B'   = Pi_H B Pi_H^T
c1'  = Pi_H c1
c2'  = Pi_H c2 Pi_H^T
D'   = Pi_H D Pi_H^T

Run the reindexer again to obtain reordered tensors:

A*', m*', B*', c1*', c2*', D*'

Then compare reordered forms, not the permutations themselves:

L_cons =
    MSE(A*,  A*')
  + MSE(m*,  m*')
  + MSE(B*,  B*')
  + MSE(c1*, c1*')
  + MSE(c2*, c2*')
  + MSE(D*,  D*')

This is the consistency principle defined in the reindexer plan.

13.3 Reindexer objective

Implement:

L_reindex = L_task + alpha_loc * L_loc + beta_cons * L_cons
13.4 Default staged coefficients

For the initial stable implementation config:

alpha_loc = 0.0
beta_cons = 0.0

Later configs:

stage_1: alpha_loc = 0.02, beta_cons = 0.00
stage_2: alpha_loc = 0.05, beta_cons = 0.10
14. Training Loop

The reindexer plan explicitly requires two optimizers and two forward passes per batch.

14.1 Files

Update or create:

training/train_label_free_v11.py
training/train_utils.py
14.2 Optimizers

Use:

optimizer_mapper    = AdamW(mapper_params,    lr=1e-4, weight_decay=1e-4)
optimizer_reindexer = AdamW(reindexer_params, lr=5e-5, weight_decay=1e-4)

The learning-rate split follows the reindexer defaults.

14.3 Gradient clipping

Use:

clip_grad_norm_(mapper_params,    1.0)
clip_grad_norm_(reindexer_params, 1.0)

The reindexer defaults clip both at 1.0.

14.4 Pass A — mapper update

For one batch:

read native tensors (A, m, B, c1, c2, D)

compute R_L, R_H

detach them:

R_L_det = R_L.detach()
R_H_det = R_H.detach()

build reordered tensors using detached permutations

build reordered hardware tokens

run mapper to get S*

decode S_nat

compute P_map

compute L_task

backward only through mapper

clip mapper gradients

optimizer step for mapper only

14.5 Pass B — reindexer update

recompute R_L, R_H without detach

rebuild reordered tensors

rebuild reordered hardware tokens

run mapper with mapper parameters frozen

decode S_nat

compute P_map

compute L_task

compute L_loc, L_cons

compute L_reindex

backward only through reindexer

clip reindexer gradients

optimizer step for reindexer only

This follows the exact optimization separation prescribed in the reindexer plan.

14.6 Temperature scheduling

Use:

tau_r:
    start = 1.0
    end   = 0.15
    schedule = cosine or linear anneal over total training steps

Mapper Sinkhorn temperature:

tau_m = fixed

Do not anneal tau_m in v1.1 unless you already have a stable implementation doing so.

14.7 Batch size

Use:

batch_size = 1

Always.

15. Inference Protocol

The reindexer plan specifies hard reindexing at inference.

15.1 Steps

For one circuit:

extract native (A, m, B, c1, c2, D)

compute reindexer logits G_L_tilde, G_H_tilde

compute hard permutations:

R_L_hat = Hungarian(G_L_tilde)
R_H_hat = Hungarian(G_H_tilde)

reorder tensors with hard permutations

build hardware tokens from hard-reordered hardware tensors

run mapper to get S*

decode:

S_nat = R_L_hat.T @ S* @ R_H_hat

run Hungarian on S_nat:

M_map = Hungarian(S_nat)

return mapping for rows u < K

15.2 Output format

Return:

{
  "mapping": {logical_u: native_physical_j, ...},
  "M_map": M_map,
  "S_nat": S_nat,
  "R_L_hat": R_L_hat,
  "R_H_hat": R_H_hat
}
16. Validation and Fail-Safes

This is mandatory, not optional.

16.1 Tensor checks

After each major tensor is created, assert:

correct shape,

finite values only,

no NaN,

no Inf.

Apply to:

A, m, B, c1, c2, D
R_L, R_H
A*, m*, B*, c1*, c2*, D*
S*, S_nat
P_map
16.2 Loss checks

Log and validate each of:

L_PST_1Q
L_PST_2Q
L_PST_total
L_swap
L_depth
L_task
L_loc_log
L_loc_hw
L_loc
L_cons
L_reindex
16.3 Sinkhorn sanity checks

For each Sinkhorn output:

row sums,

column sums,

min,

max,

entropy/sharpness.

Apply to:

R_L
R_H
P_map
16.4 Gradient checks

After backward, log:

mapper grad norm,

reindexer grad norm,

whether grad is finite.

16.5 Hard-fail conditions

Abort the step immediately if any of the following occurs:

any major tensor contains NaN/Inf,

any loss term is non-finite,

any gradient norm is non-finite,

any Sinkhorn output is non-finite.

16.6 Warning conditions

Warn but continue if:

max abs of a tensor exceeds 1e6,

Sinkhorn row/column sums deviate by more than 1e-2 from expected.

16.7 Failure report

On failure, log:

circuit_id
dataset_source
K
global_step
epoch
pass_type
offending_tensor_or_loss
min/max/mean
tau_r
tau_m
mapper_lr
reindexer_lr
17. File and Module Layout

Recommended file structure:

mqm/
  networks/
    unet_mapper.py
    reindexer.py
    hardware_token_encoder.py
    assignment.py

training/
  backend_extractor.py
  circuit_featurizer.py
  dataset_graphqmap.py
  loss_task.py
  loss_reindexer.py
  train_label_free_v11.py
  evaluate.py
  samplers.py
  checks.py
  config_v11.py
17.1 backend_extractor.py

Responsibilities:

extract B_raw, c1_raw, c2_raw

compute D_raw

normalize to B, c1, c2, D

17.2 circuit_featurizer.py

Responsibilities:

parse circuit

compute A_raw

compute m

normalize to A

17.3 dataset_graphqmap.py

Responsibilities:

load QASM

parse circuit

filter by K <= 27

reject degenerate circuits

return native tensors only:

A, m, B, c1, c2, D, metadata

This native-only dataset output is required by the reindexer design.

17.4 reindexer.py

Responsibilities:

implement reindexer branches

implement log-Sinkhorn

produce R_L, R_H

optionally expose hard permutation inference helper

17.5 hardware_token_encoder.py

Responsibilities:

build x_hw[j]

embed to T_hw

17.6 unet_mapper.py

Responsibilities:

accept A* and T_hw*

run cross-attention at the 3 locked points

output S*

17.7 assignment.py

Responsibilities:

Sinkhorn for mapper training

Hungarian wrapper for inference

17.8 loss_task.py

Responsibilities:

L_PST_1Q

L_PST_2Q

L_PST_total

L_swap

L_depth

L_task

17.9 loss_reindexer.py

Responsibilities:

L_loc_log

L_loc_hw

L_loc

L_cons

L_reindex

17.10 samplers.py

Responsibilities:

MQT-only smoke sampler

balanced 3-source sampler for main training

17.11 checks.py

Responsibilities:

tensor sanity checks

Sinkhorn checks

gradient checks

structured crash reports

18. Default Config

Use this as the initial config_v11.py or YAML equivalent.

hardware:
  n_qubits: 27
  unreachable_distance_fill: 28

normalization:
  eps: 1.0e-8

dataset:
  phase_a_sources: ["mqt"]
  phase_b_sources: ["queko", "mlqd", "mqt"]
  test_sources: ["qasmbench", "revlib"]
  min_qubits: 2
  max_qubits: 27
  require_two_qubit_gate: true
  drop_degenerate_zero_interaction: true
  balance_phase_b_sources: true

reindexer:
  hidden_dim: 128
  sinkhorn_iters: 20
  tau_start: 1.0
  tau_end: 0.15
  dropout: 0.1

mapper:
  token_dim: 128
  attn_heads: 4
  attn_gate_init: 0.0
  tau_map: 0.10
  sinkhorn_iters: 20

loss:
  alpha_diag: 0.25
  beta_diag: 1.0
  lambda_p: 1.0
  lambda_s: 1.0
  lambda_d: 0.25
  kappa_depth: 1.0

auxiliary_losses:
  alpha_loc: 0.0
  beta_cons: 0.0

optim:
  batch_size: 1
  mapper_lr: 1.0e-4
  reindexer_lr: 5.0e-5
  weight_decay: 1.0e-4
  grad_clip: 1.0

training:
  hard_fail_on_nonfinite: true
  warn_on_large_abs_tensor: 1.0e6
  warn_on_sinkhorn_sum_error: 1.0e-2
19. Implementation Order

Implement in this exact order.

Phase 1 — data and preprocessing

backend extractor

circuit featurizer

dataset filtering and metadata

normalization

sanity-check scripts

Phase 2 — core reindexer

log-Sinkhorn

logical branch

hardware branch

reordered tensor construction

hard-permutation inference helper

Phase 3 — mapper integration

hardware token encoder

mapper input contract change

remove logical tokens from mapper

cross-attention K/V from hardware tokens only

output S*

Phase 4 — assignment and losses

decode to native frame

Sinkhorn/Hungarian mapping head

task loss terms

logging

failure checks

Phase 5 — training loop

two optimizers

mapper pass

reindexer pass

smoke-run config

checkpointing

Phase 6 — evaluation

inference with hard reindexing

dataset-wise evaluation

final generalization tests

20. Acceptance Criteria

The implementation is acceptable only if all are true.

20.1 Data

all dataset samples returned by the loader satisfy 2 <= K <= 27

no sample returned to training has zero 2Q interaction mass

20.2 Numerics

no inf or nan in any major tensor

smoke run completes without exploding loss

both training passes run end-to-end

20.3 Shape contract

A input to mapper is (Bsz, 1, 27, 27)

mapper token input is (Bsz, 27, 128)

mapper output is (Bsz, 1, 27, 27)

20.4 Assignment

training uses Sinkhorn on S_nat

inference uses Hungarian on S_nat

returned mapping uses native hardware IDs only

20.5 Logging

all required loss components and diagnostics are recorded per step

20.6 Reindexer

R_L and R_H are approximately doubly stochastic during training

hard reindexing works in inference

21. Explicitly Deferred to Later Revisions

These are intentionally not part of v1.1 implementation:

richer gate-type weighting for A,

extra circuit channels beyond A,

including D in mapper hardware tokens,

full activation of reindexer auxiliary losses in the first stable run,

direction-aware c2 modeling beyond the symmetric reduction,

using GraphQMap labels as supervision,

multi-backend or variable-hardware training.

22. One-Paragraph Execution Summary

Implement a native-frame label-free mapper for a fixed 27-qubit IBM backend. Extract native hardware tensors B, c1, c2, D, featurize each circuit into a fixed 27 x 27 circuit matrix A plus mask m, learn soft logical and hardware reorderings with a two-branch reindexer, run the mapper on reordered A* with hardware-only cross-attention tokens built from (B*, c2*, c1*), decode logits back to native frame, solve the training assignment with Sinkhorn and inference assignment with Hungarian, optimize PST/SWAP/depth proxy loss in the native frame, and train in two passes with strict finite-value checks and batch size 1. This preserves the original mapping-head/task-loss structure while replacing the old canonical-indexing pipeline with soft learned reindexing.