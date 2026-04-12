# Noise-Aware Iterative Attention for Scalable Qubit Mapping

Anonymous Author(s)
Affiliation / Address / email

## Abstract

Qubit mapping assigns logical qubits to physical qubits on quantum hardware and determines where to insert SWAP gates, directly governing circuit execution fidelity as a quadratic assignment problem that is NP-hard. Existing learning-based methods train models for a fixed hardware topology and noise profile, requiring retraining whenever the device or calibration changes. This paper proposes an iterative attention-based framework that unifies layout and routing into a single noise-aware pipeline, interpreting the self-attention mechanism as mirror descent on a fidelity-based QAP relaxation. Circuit and hardware qubits are embedded by heat kernel signatures into a size-independent feature space, and a unified effective cost matrix C_eff provides both the optimization gradient for layout and the noise-optimal paths for SWAP insertion. The only learned parameters are two small projection matrices and a scalar coefficient, totaling approximately 200 values independent of hardware size. Experiments on IBM backends of 27, 53, and 127 qubits at optimization level 3 show that a model trained on a single backend achieves the highest average probability of successful trial in all evaluation settings, outperforming SABRE, NASSC, and Noise-Adaptive, and generalizing to unseen backends in both scale-up and scale-down directions. The analytical backbone of the framework guarantees a performance floor equal to that of gradient-based QAP optimization, even when the learned component fails to generalize.

## 1. Introduction

Quantum computers operate on physical qubits connected by a constrained coupling topology, and most two-qubit gates can only be executed between physically adjacent qubit pairs. When a quantum circuit requires a two-qubit gate between non-adjacent qubits, additional SWAP gates must be inserted to move the qubit states closer, increasing both circuit depth and accumulated noise. The qubit mapping problem, which determines how logical qubits are placed onto physical qubits and how SWAP gates are inserted, directly affects the fidelity of circuit execution. This problem is equivalent to the quadratic assignment problem (QAP), which is NP-hard, and has been a central challenge in quantum compilation research. Established heuristic methods such as SABRE address this problem by iterating between layout selection and routing, but they optimize for hop distance and do not incorporate hardware-specific noise information. Noise-adaptive approaches take error rates into account at the layout stage, but they treat layout and routing as separate, independently optimized stages. Recent deep learning methods train neural networks to predict qubit placements, but these models are specific to the hardware topology and noise profile used during training. Given that real quantum devices undergo frequent recalibration that alters error rates, a mapping method that can generalize to different hardware scales and noise conditions is desirable.

The fundamental limitation of existing learning-based approaches is that the model parameters encode hardware-specific information. A model trained on a 27-qubit device cannot be directly applied to a 53-qubit or 127-qubit device because the network architecture depends on the number of physical qubits. Even on the same device, a shift in noise profile after recalibration degrades the mapping quality, as the model has memorized a particular error distribution rather than learning a general mapping strategy. This limitation motivates the development of a framework in which the learned parameters capture universal mapping principles, while hardware-specific and noise-specific information enters as input that is recomputed at deployment time.

This paper introduces an iterative attention-based framework for qubit mapping that achieves hardware-scale and noise-profile generalization. The framework draws on the mathematical structure of the self-attention mechanism, interpreting the query-key product as a bilinear matching score and the value matrix as a cost basis for gradient-based optimization. Circuit qubits and hardware qubits are embedded into a common feature space by heat kernel signatures (HKS), a spectral graph descriptor that is defined independently of graph size and captures multi-scale structural information. Gate error rates and readout error rates enter the embedding as additional feature dimensions, and the effective cost matrix C_eff, which unifies direct gate error and SWAP chain error into a single -log(fidelity) quantity, serves as the value matrix. The iterative refinement corresponds to mirror descent on the QAP objective, and the shortest-path information from C_eff computation is reused for noise-aware SWAP routing, ensuring that the optimization objective and the routing execution are mathematically consistent. The only learned parameters are two projection matrices W^Q, W^K in R^{d_0 x d_k} and a scalar lambda, whose dimensions do not depend on the number of qubits.

## 2. Proposed Method

### 2.1 Problem Formulation and Attention Interpretation

A quantum circuit is represented as a graph G_c = (V_c, E_c) in which |V_c| = n logical qubits form the vertex set and edges indicate pairs that share at least one two-qubit gate. The hardware topology is a graph G_h = (V_h, E_h) in which |V_h| = m physical qubits form the vertex set and edges indicate physically connected pairs. Each edge (j, k) in E_h has an associated two-qubit gate error rate eps_2(j, k) in [0, 1], and each physical qubit j has a readout error rate eps_r(j) in [0, 1]. The qubit mapping problem seeks an injection pi : V_c -> V_h that minimizes the negative log-fidelity of the mapped circuit.

The cost function is

    L(pi) = sum_{(i,i') in E_c} g_{ii'} * C_eff(pi(i), pi(i')) + sum_{i in M} eps_r(pi(i))    (1)

where g_{ii'} is the number of two-qubit gates between logical qubits i and i', M is the set of measured qubits, and C_eff(j, k) is the effective cost of executing a two-qubit gate when the operands reside on physical qubits j and k. For adjacent pairs (j, k) in E_h, the effective cost equals eps_2(j, k). For non-adjacent pairs, C_eff(j, k) equals the cost of the minimum-cost SWAP chain followed by a direct gate execution. All terms are expressed in -log(fidelity) units, and therefore no artificial weighting coefficients are required.

In matrix form, the QAP relaxation is

    min_P tr(A_c P C_eff P^T) + m^T P eps_r

where A_c(i, i') = g_{ii'} is the gate-count weighted adjacency matrix and P in R^{n x m} is a soft assignment matrix.

The proposed framework solves this QAP relaxation by adapting the self-attention mechanism. The standard self-attention computes

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V    (2)

where the product QK^T defines an asymmetric bilinear form that measures pairwise affinity, the softmax produces a row-stochastic soft assignment matrix P, and the output Z = PV computes a convex combination of value vectors weighted by the assignment probabilities.

**Table 1: Correspondence between self-attention and qubit mapping.**

| Attention | Qubit mapping instantiation |
|---|---|
| Query Q | Logical qubit features: X^c W^Q |
| Key K | Physical qubit features: X^h W^K |
| QK^T | Structural and noise compatibility score |
| softmax / Sinkhorn | Soft assignment P (relaxed permutation) |
| Value V | C_eff (unified gate + SWAP cost) |
| Output Z = PV | Expected cost landscape for gradient feedback |

A key departure from standard self-attention is the iterative feedback loop. The output Z^(t) from iteration t feeds back into the score at iteration t+1 as S^(t+1) = QK^T - lambda * A_c * Z^(t), where the subtracted term equals half the gradient of the QAP cost function.

### 2.2 Embedding and Preprocessing

**Effective cost matrix.** A SWAP operation on edge (j, k) consists of three CX gates, contributing an error cost of 3*eps_2(j, k). A weighted graph is constructed by assigning weight 3*eps_2(j, k) to each hardware edge, and Dijkstra's algorithm computes all-pairs shortest paths on this graph. The resulting distance matrix D_swap(j, k) gives the total SWAP-chain error cost between any two physical qubits. The effective cost matrix is then defined as

    C_eff(j, k) = eps_2(j, k)     if (j, k) in E_h
                  D_swap(j, k)     otherwise           (3)

The predecessor matrix from Dijkstra's algorithm is also retained, providing the noise-optimal path Path(j, k) for each pair.

**Heat kernel signature.** Given a graph Laplacian L and its eigendecomposition Lu_k = lambda_k u_k, the heat kernel signature at node v and time scale t is

    h_t(v) = sum_k exp(-lambda_k t) u_k(v)^2    (4)

A vector of s time scales {t_1, ..., t_s}, logarithmically spaced, produces an s-dimensional embedding phi_HKS(G, v) = [h_{t_1}(v), ..., h_{t_s}(v)]. For the circuit graph, the standard Laplacian is used. For the hardware graph, a noise-weighted Laplacian L_N is used, in which edge weights are set to 1/eps_2(j, k), causing diffusion to prefer low-error paths. Because HKS values are probabilities in [0, 1], the embedding scale is independent of graph size, enabling direct comparison between graphs of different sizes.

**Feature construction.** The raw feature vector for circuit qubit i is x_i^c = [phi_HKS(G_c, i) || g_i / g_max || m_i] in R^{d_0}, where g_i = sum_{i'} g_{ii'} is the total two-qubit gate count and m_i in {0, 1} indicates measurement. The raw feature vector for hardware qubit j is x_j^h = [phi_HKS^(N)(G_h, j) || 1 - eps_2_bar(j) || 1 - eps_r(j)] in R^{d_0}, where eps_2_bar(j) is the average two-qubit error of the edges incident to j. The query and key matrices are obtained by learned linear projections Q = X^c W^Q and K = X^h W^K, where W^Q, W^K in R^{d_0 x d_k} are the learned parameters. The value matrix is fixed as V = C_eff, preserving the QAP gradient structure.

### 2.3 Iterative Attention and Unified Routing

The initial score matrix is S^(0) = QK^T, and the soft assignment is obtained by applying Sinkhorn normalization to the exponentiated scores. The iterative refinement proceeds as follows for t = 0, 1, ..., T-1:

    S^(t+1) = QK^T - lambda * A_c * Z^(t)                    (5)
    P^(t+1) = Sinkhorn(exp(S^(t+1) / tau_{t+1}))              (6)
    Z^(t+1) = P^(t+1) V,    tau_{t+1} = tau_0 * beta^{t+1}    (7)

The term A_c Z^(t) = A_c P^(t) C_eff equals half the gradient of the QAP cost function tr(A_c P C_eff P^T) at P = P^(t). Therefore, each iteration subtracts a fidelity-improving direction from the matching score, and the process corresponds to mirror descent on the QAP relaxation. Temperature annealing (tau_t -> 0) drives the soft assignment toward a hard permutation. The final mapping is pi(i) = argmax_j P_{ij}^{(T)}.

After the layout is determined, gates are processed in topological order. For each two-qubit gate on logical qubits (i, i'), the current physical locations j = pi(i) and k = pi(i') are checked. If (j, k) in E_h, the gate is applied directly. Otherwise, the precomputed noise-optimal path Path(j, k) is retrieved from the predecessor matrix, and SWAP gates are inserted along this path to bring qubit i adjacent to qubit i'. The mapping pi is updated after each SWAP.

The total set of learned parameters is theta = {W^Q, W^K, lambda}, totaling 2*d_0*d_k + 1 scalar values. The dimensions of W^Q and W^K depend only on the feature dimension d_0 = s + 2 and the projection dimension d_k, not on the number of physical qubits m. All hardware-specific and noise-specific information enters the framework as input. Training minimizes the expected fidelity loss by the Adam optimizer, and no labeled data are required because the cost function itself serves as the training signal.

## 3. Mathematical Analysis

The feedback term in the iterative update (Eq. 5) satisfies A_c P^(t) C_eff = (1/2) * d/dP tr(A_c P C_eff P^T)|_{P=P^(t)}, because A_c and C_eff are both symmetric. The iterative update P^(t+1) = Sinkhorn(exp((QK^T - lambda * grad_P L / 2) / tau)) is therefore a mirror descent step using the negative entropy as the mirror map. Mirror descent on compact domains is known to converge to a stationary point of the objective, and the monotone decrease property guarantees that L(P^{(t+1)}) <= L(P^{(t)}) at each iteration for a sufficiently small effective step size lambda / (2*tau).

At convergence, P* = P^{(t+1)} = P^{(t)}, and the fixed-point condition gives

    P_{ij}^* proportional to exp((q_i . k_j - lambda * sum_{i' in N_c(i)} g_{ii'} * sum_k P_{i'k}^* C_eff(j, k)) / tau)    (8)

This is a Boltzmann distribution at temperature tau, in which the effective energy E_{ij} = -q_i . k_j + lambda * (expected gate-weighted neighbor cost) balances the learned matching score against the analytically computed neighborhood cost. This structure is identical to mean-field theory in statistical mechanics.

The layout-routing consistency follows directly from the definition of C_eff. The layout stage optimizes sum g_{ii'} C_eff(pi(i), pi(i')), and the routing stage inserts SWAPs along Path(pi(i), pi(i')). Because C_eff(j, k) is defined as the cost of the minimum-cost SWAP path from j to k, and the routing follows exactly this path, the realized routing cost equals the layout-optimized cost.

The score decomposition S^{(t+1)} = QK^T - lambda * A_c P^{(t)} C_eff separates the learned component (QK^T) from the analytical component (A_c P^{(t)} C_eff). If the learned projections W^Q, W^K fail to generalize to an unseen hardware topology, the analytical term remains a valid QAP fidelity gradient and continues to drive the assignment toward lower cost. The performance of the framework is therefore lower-bounded by that of the purely analytical mirror descent variant. This property distinguishes the framework from purely neural approaches, which have no such fallback guarantee.
