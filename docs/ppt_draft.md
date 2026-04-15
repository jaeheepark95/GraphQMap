# GraphQMap PPT 초안 — 교수님 첫 미팅

> 슬라이드별 내용 정리. 각 `## Slide N`이 한 장의 슬라이드에 해당.

---

## Slide 1: Title

**GraphQMap: Hardware-Agnostic Quantum Qubit Mapping via Graph Neural Networks**

- 이름, 소속, 날짜

---

## Slide 2: Background — Quantum Circuit Compilation

- 양자 회로는 논리 큐빗(logical qubit)으로 작성됨
- 실제 NISQ 하드웨어에서 실행하려면 물리 큐빗(physical qubit)에 매핑해야 함
- 하드웨어 제약:
  - 모든 큐빗 쌍이 연결되어 있지 않음 (limited connectivity)
  - 큐빗마다 noise 수준이 다름 (T1, T2, gate error, readout error)
- 연결되지 않은 큐빗 간 2-qubit gate 실행 시 SWAP gate 삽입 필요 → 회로 깊이 증가 → 오류 누적

---

## Slide 3: Background — Initial Layout의 중요성

- Quantum Circuit Compilation = **Initial Layout** + Routing + Optimization + Scheduling
- Initial Layout: 어떤 논리 큐빗을 어떤 물리 큐빗에 배치할 것인가
- Layout이 좋으면 → SWAP 수 감소 → 회로 깊이 감소 → PST(성공 확률) 향상
- Layout이 나쁘면 → SWAP 폭증 → noise 누적 → 실행 결과 무의미
- **같은 회로, 같은 하드웨어에서도 layout에 따라 PST가 크게 달라짐**

---

## Slide 4: Background — 기존 방법의 한계

| 방법 | 접근 | 한계 |
|------|------|------|
| SABRE | Heuristic (SWAP 횟수 최소화) | Noise 고려 안 함 |
| Noise-Adaptive | Noise 기반 subgraph isomorphism | 회로 구조 고려 부족 |
| NASSC | Noise + SWAP 비용 통합 | Hardware-specific 튜닝 필요 |
| QAP | Quadratic Assignment 최적화 | 확장성 제한 |
| OLSQ2 | SMT Solver (layout+routing 동시 최적화) | 계산 비용 매우 높음, 실시간 불가 |

- 공통 한계: **특정 하드웨어에 종속** → 새 하드웨어마다 재설정/재계산 필요
- Research Gap: 단일 모델로 다양한 하드웨어에 범용 동작하는 학습 기반 매핑 부재

---

## Slide 5: Research Objective

**목표:** 하나의 GNN 모델이 다양한 양자 하드웨어에서 high-quality initial layout을 출력

- Hardware-agnostic: hardware-specific fine-tuning 없이 동작
- Noise-aware: 하드웨어 noise 특성을 피처로 활용
- Fast inference: solver 대비 실시간 추론 가능
- 모델은 **initial layout만 담당**, routing 이후는 Qiskit SABRE에 위임

**Primary Metric:** PST (Probability of Successful Trials)

---

## Slide 6: Core Pipeline Overview

```
Input: Quantum Circuit (.qasm) + Target Hardware (FakeBackendV2)
                        |
    +-------------------+-------------------+
    |                                       |
[Circuit Graph 구성]                [Hardware Graph 구성]
    |                                       |
[Circuit GNN Encoder]              [Hardware GNN Encoder]
    |                                       |
    C (l x d)                          H (h x d)
    +-------------------+-------------------+
                        |
              [Cross-Attention Module]
                        |
                C' (l x d), H' (h x d)
                        |
                   [Score Head]
                   S (l x h)
                        |
                 [Dummy Padding]
                   S (h x h)
                        |
              [Log-domain Sinkhorn]
                   P (h x h)
                        |
              [Hungarian Algorithm]
                        |
            Initial Layout (l -> p mapping)
                        |
          [qiskit transpile + SABRE routing]
                        |
              [Noise Simulation → PST]
```

- Sinkhorn까지는 미분 가능, transpile 이후는 **non-differentiable barrier**
- 이 구조적 제약이 전체 학습 전략을 결정함

---

## Slide 7: Graph Representation — Circuit Graph

- **노드:** 각 논리 큐빗 = 1개 노드
- **엣지:** 2-qubit gate로 연결된 큐빗 쌍 (undirected, 중복 merge)

| Node Feature | 설명 |
|-------------|------|
| gate_count | 해당 큐빗의 총 게이트 수 |
| two_qubit_gate_count | 2-qubit gate 수 |
| degree | 상호작용하는 다른 큐빗 수 |
| circuit_depth_participation | 회로 depth 중 활성 비율 |

| Edge Feature | 설명 |
|-------------|------|
| interaction_count | 해당 큐빗 쌍의 2-qubit gate 수 |
| earliest_interaction | 첫 상호작용 시점 (normalized) |
| latest_interaction | 마지막 상호작용 시점 (normalized) |

- 정규화: 각 회로 내에서 z-score 정규화 (상대적 중요도)

---

## Slide 8: Graph Representation — Hardware Graph

- **노드:** 각 물리 큐빗 = 1개 노드
- **엣지:** Coupling map 연결 (undirected)

| Node Feature | 설명 | 방향 |
|-------------|------|------|
| T1 | 에너지 완화 시간 | 높을수록 좋음 |
| T2 | 위상 완화 시간 | 높을수록 좋음 |
| frequency | 큐빗 주파수 | 구조적 정보 |
| readout_error | 측정 오류율 | 낮을수록 좋음 |
| single_qubit_error | 단일 큐빗 게이트 오류 | 낮을수록 좋음 |
| degree | connectivity 차수 | 구조적 정보 |

| Edge Feature | 설명 |
|-------------|------|
| cx_error | 2-qubit gate 오류율 |
| cx_duration | 2-qubit gate 실행 시간 |

- 정규화: 각 백엔드 내에서 z-score 정규화 → cross-hardware 일반화 가능

---

## Slide 9: Model Architecture — Dual GNN Encoder

- **GATv2 (Graph Attention Network v2)** 3-layer
- Circuit GNN, Hardware GNN: 동일 구조, 별도 파라미터 (no weight sharing)

```
Input Features → Linear(input_dim, 64) → projection
  → GATv2 Layer 1 → BatchNorm → ELU → Residual
  → GATv2 Layer 2 → BatchNorm → ELU → Residual
  → GATv2 Layer 3 → BatchNorm → ELU → Residual
  → Linear(64, 64) → Final Embedding
```

- Attention Head 4개, Embedding dim 64
- Edge feature를 attention score 계산에 통합
- 3-hop 이웃 정보 aggregation

---

## Slide 10: Model Architecture — Cross-Attention

**목적:** 단순 dot-product는 "유사한" 큐빗 매칭만 포착. 실제로는 **상보적** 관계 필요
- 예: 바쁜 논리 큐빗 → 고품질 물리 큐빗 (유사하지 않지만 좋은 매핑)

**구조 (2 layers, bidirectional):**

```
Layer n:
  C = LayerNorm(C + CrossAttn(Q=C, K=H, V=H))   // 회로가 하드웨어 참조
  C = LayerNorm(C + FFN(C))

  H = LayerNorm(H + CrossAttn(Q=H, K=C, V=C))   // 하드웨어가 회로 참조
  H = LayerNorm(H + FFN(H))
```

- 회로와 하드웨어 임베딩이 서로를 참조하며 매핑에 적합한 표현으로 변환

---

## Slide 11: Model Architecture — Score Head, Sinkhorn, Hungarian

**Score Head:**
- S_ij = (C'_i * W_q)^T * (H'_j * W_k) / sqrt(d_k)
- 각 (논리 큐빗, 물리 큐빗) 쌍의 매핑 적합도 점수

**Dummy Padding:**
- l < h 이므로 (h-l)개 dummy row 추가 → h x h 정방 행렬

**Log-domain Sinkhorn:**
- Score matrix → doubly stochastic matrix (각 행/열 합 = 1)
- 온도 파라미터 tau로 sharpness 제어
- Log-domain 구현 (low tau에서 수치 안정성)

**Hungarian Algorithm (추론 시만):**
- Doubly stochastic matrix → discrete one-to-one 매핑
- 상위 l개 행만 사용, dummy 할당은 폐기

---

## Slide 12: Training Strategy Overview

**Unsupervised Surrogate Training**

```
L_adj + beta * L_hop + alpha * L_node
(label 불필요, noise-aware 최적화, all data)
```

- Non-differentiable barrier를 surrogate loss로 우회

---

## Slide 14: Unsupervised Surrogate Training

**목표:** Label 없이 NISQ PST와 상관되는 surrogate loss로 fine-tuning

**데이터:** 6,887 전체 회로 (label 불필요)

**Loss = L_adj + beta * L_hop + alpha * L_node**

**L_surr (Error-Aware Edge Quality):**
- 논리적으로 연결된 큐빗 쌍이 물리적으로도 가까운 (low error path) 위치에 매핑되도록
- d_error(p,q) = cx_error weighted shortest path (Floyd-Warshall로 사전 계산)

**L_node (NISQ Node Quality):**
- 바쁜 논리 큐빗 → 고품질 물리 큐빗으로 유도
- q_score: learnable weighted combination of noise features
- 학습 후 w1~w5 분석 → 어떤 noise factor가 PST에 중요한지 insight

**Multi-programming:** 여러 회로를 하나의 disconnected circuit graph로 합쳐서 처리. L_adj의 intra-circuit clustering이 자연스럽게 회로 간 분리를 만듦.

---

## Slide 15: Dataset Overview

| Dataset | 회로 수 | Label |
|---------|:-------:|:-----:|
| QUEKO | 900 (540 labeled) | O |
| MLQD | 4,443 (3,729 labeled) | O |
| MQT Bench | 1,219 | X |
| QASMBench | 94 | X |
| RevLib | 231 | X |

- **총 6,887 training circuits, 4,269 labels**
- 전처리: gate normalization ({cx, id, rz, sx, x}), extreme circuit filtering, benchmark dedup

---

## Slide 16: Hardware Backends

**Training: 60 backends**
- 55 Qiskit FakeBackendV2 (5Q ~ 127Q)
- 5 synthetic backends (QUEKO/MLQD 하드웨어 topology + 합성 noise)

**Test (UNSEEN): 3 backends**
- FakeToronto (27Q)
- FakeBrooklyn (65Q)
- FakeTorino (133Q)

- 학습 시 본 적 없는 하드웨어에서 평가 → hardware-agnostic 성능 검증
- Native 2-qubit gate 종류 자동 감지 (cx / ecr / cz)

---

## Slide 17: Evaluation Protocol

**Primary Metric:** PST (Probability of Successful Trials)
- Noise simulation으로 측정 (Qiskit Aer, tensor_network + GPU)
- 8,192 shots, optimization_level 3

**Baselines:**
- SABRE, Trivial, Dense, NASSC, Noise-Adaptive, QAP

**Benchmark Circuits:** 8개 standard circuits (3Q ~ 9Q)
- toffoli_3, fredkin_3, 3_17_13, 4mod5-v1_22, mod5mils_65, alu-v0_27, decod24-v2_43, 4gt13_92

---

## Slide 18: Current Progress

**구현 완료:**
- 데이터 파이프라인 (5개 데이터셋, 전처리, split 관리)
- 모델 아키텍처 (Dual GNN, Cross-Attention, Score Head, Sinkhorn, Hungarian)
- 학습 루프 (surrogate losses, tau annealing)
- 평가 프레임워크 (PST 측정, baseline 비교, benchmark runner)
- 119 unit/integration tests

**현재 단계:**
- 학습 및 실험 진행 중
- (초기 결과 있으면 여기에 추가)

---

## Slide 19: Expected Contributions

1. **Hardware-Agnostic Mapping:** 단일 모델이 다양한 NISQ 하드웨어에서 fine-tuning 없이 동작
2. **Noise-Aware Surrogate Loss:** Non-differentiable barrier를 우회하는 미분 가능 surrogate 손실 함수 설계
3. **Cross-Attention Architecture:** Circuit-Hardware 간 상보적 관계를 포착하는 양방향 cross-attention
4. **Learnable Quality Score:** 학습된 noise factor 가중치를 통한 해석 가능한 물리 큐빗 품질 평가

---

## Slide 20: Timeline / Next Steps

- 학습 완료 및 하이퍼파라미터 튜닝
- Unseen backend (Toronto, Brooklyn, Torino)에서 PST 평가
- Ablation study (Cross-Attention 유무, L_node 기여도 등)
- 논문 작성
- (향후) Multi-programming 확장

---

## Slide 21: Q&A
