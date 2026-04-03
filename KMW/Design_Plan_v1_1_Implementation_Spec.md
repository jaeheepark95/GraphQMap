# Design Plan v1.1 - Explicit Implementation Specification

## 0. Document status

This document is a **faithful implementation specification for v1.1** of the 27-qubit U-Net initial-mapping project.

It is written from the following v1.1 sources only:

- `Design Plan v1_1.pdf`
- `extractor.py`
- `indexer.py`
- `featurizer.py`
- `layers.py`
- `unet_mapper.py`
- `dataset v1_1.py`
- `loss v1_1.py`
- `train_label_free (1).py`
- `eval_unet_mapping (1).py`
- the provided v1.1 repository tree screenshot

The newer `Combined Design Plan v1.4.md` is used **only as a formatting reference**. Its architectural choices, training policy, losses, and later amendments must **not** be imported into this v1.1 specification.

### 0.1 Fidelity rule

Where the PDF gives the architectural intent but leaves an implementation detail vague, this document follows the **actual v1.1 code**.

Where the code and the PDF differ in specificity, this document records the **actual implemented v1.1 behavior** so another engineer can reproduce the system without guesswork.

---

## 1. Project objective

### 1.1 Core task

Given:

- a quantum circuit with `K <= 27` logical qubits
- an IBM `BackendV2` device with `n = 27` physical qubits

predict an **initial one-to-one mapping** from logical qubits to physical qubits.

The model is trained **without supervised mapping labels**. Instead, it optimizes differentiable proxy objectives intended to correlate with:

- execution reliability / PST-related quality
- routing difficulty
- depth growth after routing

### 1.2 Output contract

For each circuit sample, the system produces:

1. **logit matrix** `S in R^(27 x 27)`
   - row `u` = logical slot index
   - column `j` = physical qubit index in **canonical** hardware order

2. **soft assignment** `P` during training
   - produced by Sinkhorn normalization of `S`

3. **hard assignment** `M` during inference/evaluation
   - produced by Hungarian matching on `S`

4. **final mapping list**
   - `logical_u -> native_physical_id`
   - the canonical physical index is converted back to the backend's native qubit ID using the canonical permutation `p`

### 1.3 Hard constraints

The mapping must satisfy:

- one logical qubit maps to exactly one physical qubit
- one physical qubit is assigned to at most one logical qubit
- dummy logical rows may exist when `K < 27`, but they must not create ambiguity for the active logical qubits

---

## 2. Locked system assumptions

The following are fixed in v1.1 and should not be altered if the goal is faithful reproduction:

- physical hardware size is fixed to `n = 27`
- circuits with `K > 27` are rejected
- the model works internally in a **canonical physical index space**, not native backend ID space
- logical tensors are always padded to `27 x 27`
- the U-Net always receives a `5 x 27 x 27` grid input
- token conditioning uses **logical node tokens + physical node tokens**
- assignment uses **Sinkhorn in training** and **Hungarian in inference**
- the project is written around IBM `BackendV2`
- the provided training/evaluation scripts are written around `FakeTorontoV2`

---

## 3. Repository structure for v1.1

Below is the v1.1 project structure reflected by the provided screenshot, with the v1.1-specific modules called out explicitly.

```text
MQM/
|- mqm/                                  # Core research logic
|  |- _utils/                            # Existing internal utilities
|  |  |- _log.py                         # Logging system
|  |  |- _timer.py                       # Performance timing
|  |  |- _visualization.py               # Error / layout plots
|  |
|  |- networks/                          # Neural-network architecture
|  |  |- __init__.py
|  |  |- layers.py                       # Shared token encoders + cross-attention
|  |  |- unet_mapper.py                  # U-Net backbone + assignment head
|  |
|  |- processing/                        # Data / featurization logic
|  |  |- __init__.py
|  |  |- extractor.py                    # BackendV2 tensor extraction
|  |  |- indexer.py                      # Canonical BFS reindexing
|  |  |- featurizer.py                   # Circuit -> W, m
|  |
|  |- __init__.py
|  |- _function.py                       # Original GraMA heuristic code
|  |- _function2.py                      # GraMA variant code
|
|- training/
|  |- dataset.py                         # DataLoader dataset for K <= 27 circuits
|  |- loss.py                            # Proxy PST / SWAP / depth losses
|  |- train_label_free.py                # Label-free training loop
|
|- tests2/
|  |- benchmarks/                        # QASM benchmark library
|  |- butils.py                          # Load / merge / PST helper logic
|  |- benchmark_multi.py                 # Existing benchmark scripts
|  |- test_reliability_multi.py          # Existing benchmark scripts
|  |- eval_unet_mapping.py               # Neural-mapping evaluation script
|
|- requirements.txt
|- .gitignore
```

### 3.1 Uploaded-file to repository-path mapping

The filenames uploaded in this conversation correspond to the repository paths as follows:

- `dataset v1_1.py` -> `training/dataset.py`
- `loss v1_1.py` -> `training/loss.py`
- `train_label_free (1).py` -> `training/train_label_free.py`
- `eval_unet_mapping (1).py` -> `tests2/eval_unet_mapping.py`

---

## 4. End-to-end pipeline

The v1.1 pipeline is:

```text
BackendV2
  -> extract native hardware tensors (Anat, c1nat, c2nat)
  -> canonicalize hardware into (A, c1, c2) using permutation p

Circuit qc
  -> build logical interaction tensor W
  -> build logical active-mask m

(A, c1, c2, W, m)
  -> build 5-channel U-Net grid X
  -> build logical tokens Tlog_raw
  -> build physical tokens Tphy_raw
  -> encode tokens
  -> U-Net + cross-attention
  -> logits S

Training:
  S -> Sinkhorn -> P -> proxy loss

Inference / evaluation:
  S -> Hungarian -> M -> canonical-to-native remap via p -> initial_layout for Qiskit transpile
```

---

## 5. Hardware preprocessing: `BackendV2 -> (Anat, c1nat, c2nat)`

Module: `mqm/processing/extractor.py`
Class: `BackendV2Extractor`

### 5.1 Inputs

- `backend`: IBM `BackendV2`
- fixed `n = 27`

### 5.2 Outputs in native backend index space

- `Anat in R^(27 x 27)`
- `c1nat in R^27`
- `c2nat in R^(27 x 27)`

Indices here are the backend's **native physical qubit IDs**.

### 5.3 Adjacency extraction

The extractor searches `backend.target.operation_names` in this order:

1. `'cx'`
2. `'ecr'`

The first available one is used as the system's primary 2-qubit gate.

For every supported 2-qubit gate instance `(i, j)` with non-`None` error properties:

- `Anat[i, j] = 1.0`
- `c2nat[i, j] = props.error`

This means:

- adjacency is stored in the direction exposed by the backend target
- `c2nat` is an error-like badness score, not a similarity score
- unsupported pairs remain zero

### 5.4 Per-qubit cost extraction: `c1nat`

For each physical qubit `i`, the code builds:

```text
c1nat[i] = readout_error(i) + max(one_qubit_gate_errors(i))
```

Specifically:

- readout contribution comes from `target['measure'][(i,)]`, if present
- 1-qubit error candidates are checked in this order:
  - `'sx'`
  - `'x'`
  - `'id'`
- if one or more are present, the **maximum** of those available 1Q errors is added

Therefore, v1.1 defines:

- `c1nat` as a badness score
- `c1nat` = readout badness + representative 1Q badness

### 5.5 Important implementation note

The extractor does **not** apply any log transform, temperature scaling, or hand-normalization to hardware costs. It uses the target's raw error values directly.

---

## 6. Canonical physical reindexing

Module: `mqm/processing/indexer.py`
Class: `CanonicalIndexer`

The purpose of canonical indexing is to replace backend-native qubit IDs with a deterministic internal order so the neural network always sees the hardware in a consistent 27-qubit frame.

### 6.1 Inputs

- `Anat`
- `c1nat`
- `c2nat`

all in native backend order.

### 6.2 Output permutation

The indexer produces:

- `p`: array of length 27
  - interpretation: `canonical_index -> native_id`

The inverse map is conceptually `p^{-1}: native_id -> canonical_index`, although the uploaded v1.1 code only returns `p` explicitly.

### 6.3 Undirected traversal graph

The BFS traversal uses an undirected view of the hardware graph:

```text
adj = (Anat + Anat.T > 0)
```

and then:

```text
degrees[i] = sum_j adj[i, j]
```

### 6.4 Mean edge cost used in root scoring

For each node `i`, define its undirected neighbors from `adj`.

If node `i` has at least one neighbor, then:

```text
mean_edge_cost(i) = mean_j min(c2nat[i, j], c2nat[j, i])
```

over those neighbors.

If node `i` has no neighbors, the implementation assigns:

```text
mean_edge_cost(i) = 1.0
```

as an isolation penalty.

### 6.5 Z-score function

The code uses:

```text
z(x) = (x - mean(x)) / std(x),    if std(x) > 0
z(x) = x - mean(x),               otherwise
```

### 6.6 Root score

The canonical root score is:

```text
qscore(i) = z(c1nat[i]) + z(mean_edge_cost(i)) - 0.3 * z(degree(i))
```

The initial BFS root is the unvisited node with smallest `qscore`.

### 6.7 BFS traversal and tie-breaking

Once a root is chosen, BFS visits unvisited neighbors of the current node `u` sorted by the following ascending/descending priority:

1. `min(c2nat[u, v], c2nat[v, u])` ascending
2. `c1nat[v]` ascending
3. `degree(v)` descending
4. native ID `v` ascending

If the graph is disconnected, BFS restarts from the remaining unvisited node with minimum `qscore`.

### 6.8 Canonicalized tensors

Once `p` is determined:

```text
A  = Anat[p][:, p]
c1 = c1nat[p]
c2 = c2nat[p][:, p]
```

These canonical tensors are what the model actually consumes.

### 6.9 Interpretation rule

All neural-network processing is done in canonical hardware order.

Only after inference do we convert a chosen canonical physical index `j` back to the backend-native physical qubit ID using:

```text
native_id = p[j]
```

---

## 7. Circuit featurization: `qc -> (W, m)`

Module: `mqm/processing/featurizer.py`
Class: `CircuitFeaturizer`

### 7.1 Inputs

- a quantum circuit `qc`
- fixed `n = 27`

### 7.2 Output tensors

- `W in R^(27 x 27)`
- `m in R^27`

### 7.3 Logical interaction matrix `W`

`W` is initialized as zeros.

The featurizer iterates through `circuit.data`. For every instruction that acts on exactly 2 qubits:

- let the logical indices be `u` and `v`
- update:

```text
W[u, v] += 1.0
W[v, u] += 1.0
```

Therefore:

- `W` is symmetric
- `W[u, v]` equals the count of 2-qubit interactions between logical qubits `u` and `v`
- 1-qubit gates do not contribute to `W`
- all rows/columns outside the active logical range remain zero

### 7.4 Logical mask `m`

If the circuit has `K = qc.num_qubits`, then:

```text
m[u] = 1.0    for u < K
m[u] = 0.0    for u >= K
```

This is the v1.1 active-logical indicator.

### 7.5 Rejection rule

If `K > 27`, the featurizer raises an error.

---

## 8. Dataset object and precomputed physical distance matrix

Module: `training/dataset.py`
Class: `QubitMappingDataset`

### 8.1 Dataset responsibility

The dataset does **not** recompute hardware tensors for every sample. It:

1. extracts and canonicalizes hardware once
2. precomputes the physical shortest-path distance matrix once
3. reuses those tensors for every circuit in the dataset

### 8.2 Initialization steps

On dataset construction:

1. store `circuits`
2. instantiate:
   - `CircuitFeaturizer`
   - `BackendV2Extractor`
   - `CanonicalIndexer`
3. extract `(Anat, c1nat, c2nat)` from the backend
4. compute `p`
5. canonicalize to `(A, c1, c2)`
6. build undirected graph `G = nx.from_numpy_array(A)`
7. compute all-pairs shortest-path distances using `nx.floyd_warshall_numpy(G)`
8. normalize the distance matrix to `[0, 1]` by dividing by its maximum value

The resulting normalized matrix is stored as:

```text
D in R^(27 x 27)
```

### 8.3 Per-sample return value

For a circuit at index `idx`, `__getitem__` returns a dictionary with:

- `W`: `(27, 27)`
- `m`: `(27,)`
- `A`: `(27, 27)`
- `c1`: `(27,)`
- `c2`: `(27, 27)`
- `D`: `(27, 27)`

### 8.4 Important implementation note

`A`, `c1`, `c2`, and `D` are identical for all samples in the same run because the backend is fixed. Only `W` and `m` vary per circuit.

---

## 9. Exact model inputs

The model consumes two input families:

1. a 5-channel spatial grid `X`
2. node-token feature sets `Tlog_raw` and `Tphy_raw`

### 9.1 U-Net grid input `X`

For each sample, build the following `27 x 27` channels:

1. `X0 = W`
2. `X1 = A`
3. `X2 = c2`
4. `X3 = C1_col`, where `C1_col[:, j] = c1[j]`
5. `X4 = Mmask`, where `Mmask[u, :] = m[u]`

In code, channels 4 and 5 are built as:

```python
X3 = c1.unsqueeze(1).repeat(1, 27, 1)
X4 = m.unsqueeze(2).repeat(1, 1, 27)
X  = torch.stack([W, A, c2, X3, X4], dim=1)
```

Thus:

```text
X.shape = (B, 5, 27, 27)
```

### 9.2 Logical raw token features `Tlog_raw`

Although the PDF leaves logical token statistics somewhat open-ended, the actual v1.1 implementation fixes them to exactly 3 scalars per logical slot:

```text
Tlog_raw[u] = [W_row_sum(u), 0, m[u]]
```

where:

- `W_row_sum(u) = sum_v W[u, v]`
- the second feature is a literal zero placeholder
- `m[u]` is the dummy/active indicator

In code:

```python
Tlog_raw = torch.stack([
    W.sum(dim=-1),
    torch.zeros_like(m),
    m
], dim=-1)
```

Therefore:

```text
Tlog_raw.shape = (B, 27, 3)
```

### 9.3 Physical raw token features `Tphy_raw`

The physical token has exactly 4 scalar features per canonical physical qubit `j`:

```text
Tphy_raw[j] = [
    c1[j],
    degree_A(j),
    mean(c2[j, :]),
    min(c2[j, :])
]
```

In code:

```python
Tphy_raw = torch.stack([
    c1,
    A.sum(dim=-1),
    c2.mean(dim=-1),
    c2.min(dim=-1).values
], dim=-1)
```

Thus:

```text
Tphy_raw.shape = (B, 27, 4)
```

### 9.4 Important interpretation note

The physical-token mean and min are computed over the **entire canonical row of `c2`**, not only over graph neighbors.

---

## 10. Token encoders

Module: `mqm/networks/layers.py`
Class: `SharedTokenEncoder`

### 10.1 Purpose

The token encoders map raw logical and physical node features into a shared embedding space before cross-attention conditioning.

### 10.2 Architecture

For an input feature vector of size `in_dim`, the encoder is:

```text
LayerNorm(in_dim)
-> Linear(in_dim, embed_dim)
-> ReLU
-> Dropout(0.1)
-> Linear(embed_dim, embed_dim)
-> LayerNorm(embed_dim)
```

Default embedding dimension in v1.1:

```text
embed_dim = 128
```

### 10.3 Logical vs physical encoders

The U-Net module instantiates:

- `logical_encoder = SharedTokenEncoder(in_dim=3, embed_dim=128)`
- `physical_encoder = SharedTokenEncoder(in_dim=4, embed_dim=128)`

### 10.4 Type embeddings

After encoding, v1.1 adds learned type embeddings:

- `type_embed_log` of shape `(1, 1, 128)`
- `type_embed_phy` of shape `(1, 1, 128)`

Then:

```text
Tlog = logical_encoder(Tlog_raw) + type_embed_log
Tphy = physical_encoder(Tphy_raw) + type_embed_phy
T    = concat(Tlog, Tphy, dim=1)
```

So the final token tensor is:

```text
T.shape = (B, 54, 128)
```

with the first 27 tokens representing logical slots and the next 27 representing physical nodes.

---

## 11. U-Net backbone

Module: `mqm/networks/unet_mapper.py`
Class: `UNetMapping`

### 11.1 Purpose

The U-Net backbone maps the 5-channel grid `X`, conditioned on token tensor `T`, into a `27 x 27` mapping-score matrix.

### 11.2 Exact layer stack

The implemented v1.1 backbone is:

```text
Input: X in R^(B x 5 x 27 x 27)

Down path:
- down1: Conv2d(5, 64, kernel_size=3, padding=1)
- down2: Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

Cross-attention injection:
- attn_down on the down2 feature map

Bottleneck:
- bottleneck: Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
- attn_bottleneck on the bottleneck feature map

Up path:
- up1: ConvTranspose2d(256, 128, kernel_size=2, stride=2)
- interpolate up1 output to match down2 spatial size
- attn_up on that feature map

Skip / output:
- add skip: u1 + d2
- final: Conv2d(128, 1, kernel_size=1)
- interpolate output to (27, 27)
- squeeze the channel dimension
```

### 11.3 Forward path in exact order

The code executes:

```python
d1 = relu(down1(X))
d2 = relu(down2(d1))
d2 = attn_down(d2, T)

bn = relu(bottleneck(d2))
bn = attn_bottleneck(bn, T)

u1 = relu(up1(bn))
u1 = interpolate(u1, size=d2.shape[2:])
u1 = attn_up(u1, T)

S = final(u1 + d2)
S = interpolate(S, size=(27, 27))
return S.squeeze(1)
```

### 11.4 Output tensor

The model returns:

```text
S.shape = (B, 27, 27)
```

with semantic meaning:

```text
S[b, u, j] = score for mapping logical slot u to canonical physical qubit j
```

### 11.5 No extra blocks beyond what is shown

The uploaded v1.1 code does **not** include:

- residual UNet blocks
- attention normalization blocks around the injector
- multi-stage decoder ladder beyond the single transpose-conv up block
- positional embeddings for the token sequence
- timestep embeddings
- diffusion-specific modules

If those are added, the result is no longer faithful to v1.1.

---

## 12. Cross-attention injector

Module: `mqm/networks/layers.py`
Class: `CrossAttentionInjector`

### 12.1 Injection locations

Cross-attention is used at exactly three points:

1. after `down2`
2. after `bottleneck`
3. after `up1`

### 12.2 Input and output shapes

At an injection site, the feature map has shape:

```text
F_map in R^(B x C x H x W)
```

and tokens have shape:

```text
T_tokens in R^(B x L x d)
```

with `L = 54` and `d = 128`.

### 12.3 Exact implemented mechanics

The code does the following:

1. flatten the feature map spatially

```text
F_flat = reshape(F_map) -> (B, H*W, C)
```

2. compute projections

```text
Q = q_proj(F_flat)    -> (B, H*W, d_a)
K = k_proj(T_tokens)  -> (B, L,   d_a)
V = v_proj(T_tokens)  -> (B, L,   d_a)
```

with `d_a = token_dim = 128`

3. compute scaled dot-product attention

```text
AttnWeights = softmax((Q K^T) / sqrt(d_a), dim=-1)
Out         = AttnWeights V
```

4. project back to feature-map channel size

```text
Delta_F = o_proj(Out)
```

5. reshape back to `(B, C, H, W)`

6. apply residual update with a learned scalar gate `alpha`

```text
F_map <- F_map + alpha * Delta_F
```

### 12.4 Gate initialization

Each attention injector owns a learnable scalar parameter:

```text
alpha = 0 at initialization
```

This is intended to stabilize training by letting the model start close to a pure convolutional backbone.

### 12.5 Important implementation detail

The class stores `num_heads=4`, but the uploaded v1.1 code does **not actually split into multiple heads**. The implemented behavior is therefore a **single attention computation in a 128-dimensional space**, not a true explicit multi-head decomposition.

For faithful reproduction, keep that behavior unless you intentionally want a modified version.

---

## 13. Assignment head

Module: `mqm/networks/unet_mapper.py`
Class: `AssignmentHead`

The assignment head has two modes:

- `sinkhorn` for training
- `hungarian` for hard inference

### 13.1 Sinkhorn soft assignment

Input:

```text
S in R^(B x 27 x 27)
```

The implementation computes:

```python
S_norm = (S - rowwise_max(S)) / tau
P = exp(S_norm)
for _ in range(iterations):
    P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)
    P = P / (P.sum(dim=-2, keepdim=True) + 1e-12)
```

Important details:

- the stabilizing max subtraction is done rowwise along the last axis
- there is no log-domain Sinkhorn
- normalization alternates row normalization then column normalization
- epsilon is `1e-12`

Default method signature:

```text
sinkhorn(S, tau=0.5, iterations=20)
```

Actual training call in `train_label_free.py`:

```text
tau = 0.5
iterations = 30
```

### 13.2 Hungarian hard assignment

The Hungarian implementation:

1. converts the score matrix to NumPy
2. replaces non-finite values with large finite sentinels
3. adds small Gaussian tie-breaking noise of magnitude `1e-9`
4. runs `scipy.optimize.linear_sum_assignment(..., maximize=True)`
5. writes the selected pairs into a one-hot matrix `M`

Thus:

```text
M.shape = (B, 27, 27)
M[b, u, j] in {0, 1}
```

### 13.3 Interpretation rule

The Hungarian output is a full 27-by-27 permutation-style matrix. Only the first `K` logical rows correspond to real logical qubits.

---

## 14. Loss function

Module: `training/loss.py`
Class: `MappingProxyLoss`

### 14.1 Inputs

The loss receives:

- `P`: soft assignment matrix `(B, 27, 27)`
- `W`: logical interaction matrix `(B, 27, 27)`
- `c1`: per-qubit cost `(B, 27)`
- `c2`: per-edge cost `(B, 27, 27)`
- `D`: normalized physical shortest-path distances `(B, 27, 27)`
- `m`: logical active mask `(B, 27)`

### 14.2 Scalar hyperparameters

Default constructor arguments are:

```text
lambda_p = 1.0
lambda_s = 0.1
lambda_d = 0.1
kappa    = 1.0
```

### 14.3 PST surrogate term

The code decomposes the PST surrogate into a 1Q component and a 2Q component.

#### 14.3.1 1Q component

```text
loss_1q = sum_{u, j} P[u, j] * c1[j]
```

implemented batchwise as:

```python
loss_1q = torch.sum(P * c1.unsqueeze(1), dim=(1, 2))
```

#### 14.3.2 2Q component

First compute:

```text
P c2 P^T
```

then weight it elementwise by `W` and sum:

```text
loss_2q = sum_{u, v} (P c2 P^T)[u, v] * W[u, v]
```

implemented as:

```python
loss_2q = torch.sum(torch.bmm(torch.bmm(P, c2), P.transpose(1, 2)) * W, dim=(1, 2))
```

#### 14.3.3 Combined PST surrogate

The actual code sets:

```text
L_pst_sample = (loss_1q + loss_2q) * sum_u m[u]
```

That is, the active-qubit mask is applied as a **scalar multiplicative factor equal to the number of active logical qubits**, not as an elementwise row mask on `P`.

This is the implemented v1.1 behavior and should be reproduced as written if exact fidelity is required.

### 14.4 SWAP proxy term

The code first computes an induced logical-distance matrix from the physical distance matrix:

```text
logical_dist_eff = P D P^T
```

Then:

```text
L_swap_sample = sum_{u, v} logical_dist_eff[u, v] * W[u, v]
```

implemented as:

```python
logical_dist_eff = torch.bmm(torch.bmm(P, D), P.transpose(1, 2))
l_swap = torch.sum(logical_dist_eff * W, dim=(1, 2))
```

### 14.5 Depth proxy term

Depth is taken proportional to swap pressure:

```text
L_depth_sample = kappa * L_swap_sample
```

### 14.6 Final batch loss

The total loss is:

```text
L_total = lambda_p * mean(L_pst_sample)
        + lambda_s * mean(L_swap_sample)
        + lambda_d * mean(L_depth_sample)
```

implemented as:

```python
total_loss = (
    self.lp * l_pst.mean() +
    self.ls * l_swap.mean() +
    self.ld * l_depth.mean()
)
```

---

## 15. Training procedure

Module: `training/train_label_free.py`
Function: `train_label_free`

### 15.1 Training data source

The experiment loader searches:

```text
tests2/benchmarks/*.qasm
```

Each file is loaded with `QuantumCircuit.from_qasm_file`.

Only circuits satisfying `qc.num_qubits <= 27` are kept.

### 15.2 Model and optimizer setup

The training script instantiates:

- `model = UNetMapping()`
- `loss_fn = MappingProxyLoss(lambda_p=1.0, lambda_s=0.1, lambda_d=0.1)`
- `optimizer = Adam(model.parameters(), lr=1e-4)`
- `scheduler = ReduceLROnPlateau(mode='min', patience=5, factor=0.5)`

Default training call:

```text
epochs = 50
batch_size = 16
```

### 15.3 Per-batch input assembly

For each batch dictionary:

```python
W, m, A, c1, c2, D = [v.to(device) for v in batch.values()]
```

Then construct:

```python
X3 = c1.unsqueeze(1).repeat(1, 27, 1)
X4 = m.unsqueeze(2).repeat(1, 1, 27)
X  = torch.stack([W, A, c2, X3, X4], dim=1)

Tlog_raw = torch.stack([W.sum(dim=-1), torch.zeros_like(m), m], dim=-1)
Tphy_raw = torch.stack([c1, A.sum(dim=-1), c2.mean(dim=-1), c2.min(dim=-1).values], dim=-1)
```

### 15.4 Forward and assignment

The training step performs:

```python
logits = model(X, Tlog_raw, Tphy_raw)
P = AssignmentHead.sinkhorn(logits, tau=0.5, iterations=30)
loss = loss_fn(P, W, c1, c2, D, m)
```

### 15.5 Stability measures present in the code

The script includes the following training safeguards:

1. if `loss` is `NaN`, skip the batch
2. apply gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. use the gate-`alpha=0` attention initialization from the model definition
4. use plateau-based LR reduction

### 15.6 Epoch reporting and checkpoint

At the end of each epoch:

- average batch loss is computed
- the LR scheduler is stepped on that epoch-average loss
- the script prints the epoch loss and current LR

After training, the experiment script saves:

```text
unet_toronto_final.pth
```

---

## 16. Evaluation procedure

Module: `tests2/eval_unet_mapping.py`
Function: `evaluate_unet`

### 16.1 Backend and model loading

The evaluation script uses:

- backend: `FakeTorontoV2()`
- model: `UNetMapping()`
- checkpoint preference:
  1. `unet_toronto_final.pth`
  2. fallback: `unet_toronto_model.pth`

### 16.2 Hardware preprocessing

The evaluation script reproduces the same preprocessing pipeline as training:

1. `extract_tensors(backend)`
2. `get_permutation(...)`
3. `canonicalize(...)`

### 16.3 Benchmark circuit loading

The script attempts to load benchmark:

```text
decod24-v2_43
```

If unavailable, it falls back to a simple measured 5-qubit chain circuit.

### 16.4 Inference input construction

For the selected circuit:

1. build `W, m` from `CircuitFeaturizer`
2. convert all tensors to batched torch tensors
3. build `X`, `Tlog`, and `Tphy` exactly as in training

### 16.5 Hard mapping extraction

The script computes:

```python
logits = model(X, Tlog, Tphy)
M = AssignmentHead.hungarian(logits).squeeze(0)
canonical_indices = torch.argmax(M[:K], dim=1).cpu().numpy()
mapping = {i: int(p[j]) for i, j in enumerate(canonical_indices)}
```

So the final mapping is:

```text
logical index i -> native backend qubit p[j]
```

### 16.6 Qiskit routing and PST evaluation

The circuit is transpiled with the neural mapping used as the initial layout:

```python
transpiled = transpile(
    circuit,
    backend=backend,
    initial_layout=list(mapping.values()),
    optimization_level=2
)
```

Then the evaluation script:

1. creates a noisy `AerSimulator` from the backend
2. creates an ideal simulator with `noise_model=None`
3. runs both on the transpiled circuit
4. computes `PSTv2(noisy_counts, ideal_counts)`
5. prints:
   - PST
   - circuit depth
   - gate counts

---

## 17. Exact tensor-shape summary

### 17.1 Hardware tensors

| Tensor | Shape | Meaning |
|---|---:|---|
| `Anat` | `(27, 27)` | native-ID adjacency |
| `c1nat` | `(27,)` | native-ID qubit badness |
| `c2nat` | `(27, 27)` | native-ID edge badness |
| `A` | `(27, 27)` | canonical adjacency |
| `c1` | `(27,)` | canonical qubit badness |
| `c2` | `(27, 27)` | canonical edge badness |
| `D` | `(27, 27)` | normalized canonical shortest-path distances |

### 17.2 Logical tensors

| Tensor | Shape | Meaning |
|---|---:|---|
| `W` | `(27, 27)` | symmetric logical 2Q interaction counts |
| `m` | `(27,)` | logical active-mask |

### 17.3 Model tensors

| Tensor | Shape | Meaning |
|---|---:|---|
| `X` | `(B, 5, 27, 27)` | U-Net grid input |
| `Tlog_raw` | `(B, 27, 3)` | raw logical node features |
| `Tphy_raw` | `(B, 27, 4)` | raw physical node features |
| `T` | `(B, 54, 128)` | encoded token sequence |
| `S` | `(B, 27, 27)` | mapping logits |
| `P` | `(B, 27, 27)` | soft doubly-normalized assignment |
| `M` | `(B, 27, 27)` | hard Hungarian one-hot mapping |

---

## 18. Minimal reference pseudocode

```python
# hardware preprocessing
Anat, c1nat, c2nat = extract_tensors(backend)
p = get_permutation(Anat, c1nat, c2nat)
A, c1, c2 = canonicalize(Anat, c1nat, c2nat, p)
D = normalized_shortest_path_matrix(A)

# circuit preprocessing
W, m = featurize(circuit)

# grid input
X0 = W
X1 = A
X2 = c2
X3 = repeat_columns(c1)
X4 = repeat_rows(m)
X  = stack([X0, X1, X2, X3, X4])

# token features
Tlog_raw = [row_sum(W), zeros, m]
Tphy_raw = [c1, row_sum(A), row_mean(c2), row_min(c2)]

# token encoding
Tlog = logical_encoder(Tlog_raw) + type_embed_log
Tphy = physical_encoder(Tphy_raw) + type_embed_phy
T = concat(Tlog, Tphy)

# model
S = UNet(X, T)

# training
P = sinkhorn(S, tau=0.5, iterations=30)
loss = proxy_loss(P, W, c1, c2, D, m)

# inference
M = hungarian(S)
for active logical row u in [0, ..., K-1]:
    j = argmax(M[u])
    native_physical = p[j]
```

---

## 19. Non-negotiable interpretation rules

To avoid implementation drift, the following meanings are fixed in v1.1:

1. **Rows always mean logical slots.**
   Columns always mean physical qubits.

2. **`S`, `P`, and `M` are in canonical physical order.**
   Native backend IDs appear only after remapping through `p`.

3. **`W` is a symmetric 2Q interaction-count matrix.**
   It is not a learned adjacency and not a directed gate-order matrix.

4. **Dummy logical slots are represented only by zero-padded `W` rows/cols and `m[u] = 0`.**

5. **The actual logical token feature set in v1.1 is only 3-dimensional.**
   Do not silently replace it with richer top-k statistics if the goal is faithful reproduction.

6. **The actual physical token feature set in v1.1 is only 4-dimensional.**

7. **The cross-attention code stores `num_heads`, but does not implement explicit head splitting.**

8. **The loss must match the code, not a cleaned-up reinterpretation.**
   In particular, the PST term's active-mask handling is the scalar `m.sum(dim=1)` multiplier shown in the code.

9. **The shortest-path distance matrix is computed from the canonical adjacency graph and normalized by its global maximum.**

10. **The training script uses Sinkhorn with `tau=0.5` and `iterations=30`, even though the method default is `iterations=20`.**

---

## 20. Reproduction checklist

A reimplementation is faithful to v1.1 only if it satisfies all of the following:

- uses `n = 27`
- extracts hardware from `BackendV2.target`
- chooses `cx` first, otherwise `ecr`, for 2Q extraction
- defines `c1nat` as `measure + max(sx, x, id)` error
- computes the canonical permutation using the exact qscore + BFS tie-break rules above
- builds `W` from symmetric counts of 2Q gates only
- builds the 5-channel grid exactly as specified
- uses 3 logical token features and 4 physical token features exactly as specified
- uses the shallow U-Net architecture shown here
- injects attention at down, bottleneck, and up locations only
- initializes each attention gate `alpha` to zero
- uses Sinkhorn during training and Hungarian during inference
- uses the exact proxy-loss equations described above
- remaps canonical indices back to native IDs through `p` before giving the layout to Qiskit

---

## 21. Final summary

Design Plan v1.1 is a **fixed-size 27-qubit label-free initial-mapping model** that combines:

- deterministic hardware canonicalization
- circuit interaction featurization
- a 5-channel `27 x 27` U-Net input grid
- lightweight node-token conditioning through gated cross-attention
- Sinkhorn soft assignment in training
- Hungarian discrete assignment in inference
- proxy losses based on hardware error costs and routing-distance pressure

If another engineer follows this document exactly, they should be able to reproduce the **actual v1.1 architecture and training/evaluation pipeline** without importing later v1.3/v1.4 design decisions.
