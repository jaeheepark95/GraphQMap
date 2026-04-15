# GenSpark PPT 생성 프롬프트

아래 프롬프트를 GenSpark에 입력하세요.

---

다음 지시에 따라 대학원 랩미팅용 PPT를 제작해주세요.

## 발표 상황
- **대상:** 지도교수님 (머신러닝/양자컴퓨팅에 익숙하시며, 큐빗 매핑 문제에 대한 기본 배경지식이 있으심)
- **성격:** 오랜만에 하는 연구 진행 보고 미팅 (처음이 아님). 배경 설명은 최소화하고 핵심 설계와 진행 상황에 집중
- **언어:** 한국어 기반, 기술 용어는 영어 그대로 (예: initial layout, cross-attention, surrogate loss, PST)
- **슬라이드 수:** 11장
- **톤:** 학술적이되 랩미팅 수준으로 자연스럽게. 격식적인 학회 발표는 아님

## 디자인 & 시각화 핵심 요구사항

**가장 중요:** 이 PPT는 텍스트 위주가 아니라 **연구 논문의 figure 수준의 다이어그램/도식 위주**로 제작해주세요.
- 모델 아키텍처, 파이프라인, 학습 전략 등은 반드시 **시각적 다이어그램(figure)**으로 표현. 텍스트 나열 금지.
- 다이어그램은 학술 논문에 들어가는 것처럼 **블록 다이어그램, 화살표 흐름도, 색상 구분** 등을 활용하여 한눈에 구조가 파악되도록
- 색상 팔레트: 깔끔한 학술 스타일 (파스텔 또는 muted tone). 너무 화려하지 않게.
- 슬라이드당 텍스트는 키워드/짧은 문장 수준으로 최소화. 설명은 speaker notes에 작성.
- 각 슬라이드에 speaker notes를 반드시 포함해주세요.

---

## 슬라이드 구성 (11장)

### Slide 1: Title
**GraphQMap: Hardware-Agnostic Quantum Qubit Mapping via Graph Neural Networks**
- 이름, 소속, 날짜 (2026.03.25)
- 심플한 배경.

---

### Slide 2: Research Objective & Approach

**핵심 메시지:** 단일 GNN 모델 → 임의의 양자 하드웨어에서 고품질 initial layout 출력

**시각적 구성:**
- 4가지 설계 목표를 차례대로 아이콘과 함께:
  1. **Hardware-agnostic:** hardware-specific fine-tuning 없이 다양한 백엔드에서 동작
  2. **Multi-programming:** 임의 개수의 회로를 하나의 백엔드에 동시 매핑 (총 논리 큐빗 ≤ 물리 큐빗)
  3. **High PST:** NISQ 하드웨어에서 높은 실행 성공 확률 달성
  4. **Fast inference:** solver 대비 실시간 추론 가능
- 하단: 아래의 간단한 흐름도를 배치

**하단 다이어그램 (좌→우 흐름):**
```
[GraphQMap] → Initial Layout ══ NON-DIFFERENTIABLE ══> [Qiskit Transpiler (Routing + Optimization)] → Transpiled Circuit → [Noise Simulation] → PST
                                    barrier
```
- GraphQMap 블록: 강조 색상 (모델 담당 영역)
- "NON-DIFFERENTIABLE BARRIER"를 점선 + 빨간 라벨로 표시
- Qiskit Transpiler 이후: 회색 톤 (모델 관여 안 함)
- 최종 PST를 Metric으로 강조

**Speaker Notes:** GraphQMap의 목표는 하나의 모델이 다양한 백엔드(5Q~127Q)로 학습하고, 한 번도 본 적 없는 3개 백엔드(Toronto 27Q, Brooklyn 65Q, Torino 133Q)에서 평가하는 것입니다. 모델은 initial layout만 출력하고, 이후 routing과 optimization은 Qiskit transpiler에 맡깁니다. 중요한 점은, initial layout 이후의 transpile 과정이 미분 불가능하여 PST에서 모델로 직접 gradient를 전달할 수 없다는 것입니다. 이 제약이 전체 학습 전략 설계의 핵심 동기입니다.

---

### Slide 3: Overall Pipeline (핵심 Figure)

**이 슬라이드가 PPT에서 가장 중요한 figure입니다. 논문 Figure 1 수준으로 완성도 높게 제작해주세요.**

**다이어그램 구성 (위에서 아래로 흐르는 블록 다이어그램):**

```
[입력층 - 좌우 병렬]
  Quantum Circuit (.qasm)              Target Hardware (Qiskit BackendV2)
         ↓                                      ↓
  Circuit Graph                         Hardware Graph
  (노드=논리큐빗, 엣지=2Q gate)          (노드=물리큐빗, 엣지=coupling map)
         ↓                                      ↓
  Circuit GNN Encoder                   Hardware GNN Encoder
  (GATv2 × 3 layers)                   (GATv2 × 3 layers)
         ↓                                      ↓
       C (l×d)                               H (h×d)
         ↓                                      ↓
         +──────────── 합류 ────────────────────+
                         ↓
              Cross-Attention Module
              (2 layers, bidirectional)
                         ↓
                  C' (l×d), H' (h×d)
                         ↓
                    Score Head
                    S (l×h)
                         ↓
                   Dummy Padding
                    S (h×h)
                         ↓
               Log-domain Sinkhorn
                    P (h×h)
                         ↓
               Hungarian Algorithm
                         ↓
              Initial Layout (l→p mapping)

  ═══════ NON-DIFFERENTIABLE BARRIER ═══════  ← 이 부분을 점선 + 빨간/주황 색으로 명확히 표시

                         ↓
           Qiskit Transpile + SABRE Routing
                         ↓
              Noise Simulation → PST
```

**디자인 지시:**
- 각 블록을 둥근 사각형으로, 색상으로 기능별 그룹핑:
  - 입력/그래프 구성: 연한 파랑
  - GNN 인코딩: 초록
  - Cross-Attention + Score: 보라/주황
  - Sinkhorn/Hungarian: 노랑
  - Non-differentiable 이후: 회색
- "NON-DIFFERENTIABLE BARRIER"를 굵은 점선 + 빨간 라벨로 시각적으로 명확히 분리
- 좌우 대칭 구조(Circuit/Hardware 병렬)를 살릴 것
- l = 논리 큐빗 수, h = 물리 큐빗 수, d = embedding dim(64)

**Speaker Notes:** 전체 파이프라인입니다. 회로와 하드웨어를 각각 그래프로 변환하고, 별도 GNN으로 인코딩한 뒤, Cross-Attention으로 상호 참조합니다. Score Head에서 매핑 적합도를 계산하고, Sinkhorn → Hungarian으로 최종 매핑을 결정합니다. 핵심은 중간의 non-differentiable barrier입니다. Qiskit transpile이 미분 불가능하므로 PST에서 모델로 gradient를 전달할 수 없고, 이것이 2-stage 학습 전략의 근본 원인입니다.

---

### Slide 4: Graph Representation (Circuit + Hardware를 1장에)

**좌우 분할 레이아웃으로 Circuit Graph / Hardware Graph를 한 슬라이드에:**

**왼쪽: Circuit Graph**
- 간단한 양자 회로 다이어그램(4큐빗) → 화살표 → 변환된 그래프(4노드)
- Node Features (4dim): [gate_count, 2Q_gate_count, degree, depth_participation]
  - 정규화: **회로 내 z-score** (상대적 중요도)
- Edge Features (3dim): [interaction_count, earliest_interaction, latest_interaction]
  - 정규화: **회로(또는 multi-programming 그룹) 내 z-score**

**오른쪽: Hardware Graph**
- 하드웨어 토폴로지(예: heavy-hex 일부) → 노드 색상으로 품질 표현(진한=고품질)
- Node Features (7dim): [T1, T2, readout_error, sq_error, degree, t1_cx_ratio, t2_cx_ratio]
  - 정규화: **백엔드 내 z-score**
- Edge Features (1dim): [cx_error]
  - 정규화: **백엔드 내 z-score**
- **백엔드 내 z-score 정규화가 hardware-agnostic의 핵심 메커니즘** → 별도 박스나 색상으로 강조

**Speaker Notes:** 회로는 큐빗=노드, 2Q gate=엣지로 변환하고, 하드웨어는 물리 큐빗=노드, coupling map=엣지로 변환합니다. 모든 feature에 z-score 정규화를 적용합니다. 회로는 회로(또는 multi-programming 그룹) 내에서, 하드웨어는 백엔드 내에서 정규화하여 상대적 중요도/품질로 표현합니다. 이를 통해 5Q 칩과 127Q 칩을 동일한 feature 공간에서 비교 가능하게 하며, 이것이 hardware-agnostic 일반화의 핵심 메커니즘입니다. Multi-programming 시에는 그룹 내 전체 회로를 함께 정규화하여, 복잡한 회로가 자연스럽게 배치 우선순위를 갖도록 합니다.

---

### Slide 5: Model Architecture Detail (Dual GNN + Cross-Attention + Score Head)

**이 슬라이드도 figure 위주로. 3개 컴포넌트를 하나의 다이어그램으로 통합:**

**구성 (위→아래):**

**(1) Dual GNN Encoder (상단)**
- 좌우 대칭: Circuit GNN / Hardware GNN
- 각각 내부: `Linear → [GATv2 + BatchNorm + ELU + Residual] × 3 → Linear`
- "동일 구조, 별도 파라미터 (no weight sharing)" 표시
- Embedding dim 64, Attention heads 4

**(2) Cross-Attention Module (중단)**
- C와 H 사이의 **양방향 화살표**를 명확히 표현
- Layer 구조:
  - C → Q, H → K,V: "논리 큐빗이 어떤 물리 큐빗에 주목?"
  - H → Q, C → K,V: "물리 큐빗이 어떤 논리 큐빗을 수용?"
- 2 layers 반복
- **핵심 메시지 박스: "유사성(similarity)이 아닌 상보성(complementarity) 포착"**
  - 예시: 바쁜 논리 큐빗 → 고품질 물리 큐빗 (유사하지 않지만 좋은 매핑)

**(3) Score Head → Sinkhorn (하단) + Training/Inference 분기**
- Score Head: S_ij = (C'_i · W_q)^T · (H'_j · W_k) / √d_k + bias_j
  - W_q, W_k: learned projection (d → d_k)
  - bias_j = Linear(hw_noise_features_j): 물리 큐빗별 noise-aware bias → 저오류 큐빗에 매핑 유도
- Dummy Padding → Log-domain Sinkhorn → P (h×h doubly stochastic)
- **여기서 분기 (시각적으로 명확히 표현):**
  - **Training:** P 행렬에서 바로 Loss 계산 (미분 가능)
  - **Inference:** P → Hungarian Algorithm → 이산적 1:1 매핑 (미분 불가능)

**Speaker Notes:** GATv2 기반 3-layer GNN이 각각 회로와 하드웨어를 인코딩합니다. Cross-Attention이 이 모델의 핵심인데, 단순 dot-product는 유사한 것끼리 매칭하지만, 실제로는 바쁜 논리 큐빗을 고품질 물리 큐빗에 배치하는 상보적 관계가 필요합니다. 양방향 cross-attention으로 이를 포착합니다. Score Head에서는 C'와 H'를 각각 W_q, W_k로 투영한 뒤 scaled dot-product를 계산하고, 물리 큐빗의 noise feature에서 학습한 bias를 더해 저오류 큐빗 선호를 반영합니다. 이후 Sinkhorn으로 doubly stochastic matrix를 만드는데, 학습 시에는 이 soft assignment에서 바로 loss를 계산하고, 추론 시에만 Hungarian으로 이산 매핑을 결정합니다. 이 분기 구조가 미분 가능한 학습과 이산적 추론을 양립시키는 핵심입니다.

---

### Slide 5-1: Key Equations

**각 모듈의 핵심 수식을 간결하게 정리. Slide 5의 다이어그램 구조와 대응시켜 표현.**

**(1) GATv2 — Edge-aware Graph Attention**
```
e_ij = LeakyReLU(a^T · [W·h_i ∥ W·h_j ∥ W_e·edge_ij])
α_ij = softmax_j(e_ij)
h_i' = Σ_j α_ij · V·h_j
```
- Edge feature가 attention score에 직접 반영
- 3-layer → 3-hop neighborhood 정보 집약

**(2) Bidirectional Cross-Attention (× 2 layers)**
```
C = LayerNorm(C + CrossAttn(Q=C, K=H, V=H))
C = LayerNorm(C + FFN(C))
H = LayerNorm(H + CrossAttn(Q=H, K=C, V=C))
H = LayerNorm(H + FFN(H))
```
- 회로↔하드웨어 임베딩이 서로를 참조하여 매핑에 적합한 표현으로 변환

**(3) Score Head + Noise-Aware Bias**
```
S_ij = (C'_i · W_q)^T · (H'_j · W_k) / √d_k + bias_j
bias_j = Linear(hw_noise_features_j)
```

**(4) Sinkhorn → Assignment**
```
P = LogDomainSinkhorn(S / τ)     — doubly stochastic (h×h)
layout = Hungarian(P)             — discrete 1:1 mapping (inference only)
```

**Speaker Notes:** 각 모듈의 핵심 수식입니다. GATv2는 edge feature를 attention score에 직접 반영하여 연결의 중요도를 학습합니다. Cross-Attention은 양방향으로 2 layer 반복하며 회로와 하드웨어 임베딩을 상호 참조시킵니다. Score Head는 learned projection의 scaled dot-product에 물리 큐빗별 noise bias를 더해 매핑 적합도를 계산합니다. 마지막으로 Sinkhorn이 score를 doubly stochastic matrix로 변환하고, 추론 시 Hungarian이 이산 매핑을 결정합니다.

---

### Slide 6: Training Strategy

```
┌──────────────────────────────────────────────┐
│  Unsupervised Surrogate Training             │
│                                              │
│  전체 6,887 회로 (label 불필요)               │
│  L = L_surr + α·L_node                        │
│  • L_surr: error-weighted 최단경로 거리 최소화 │
│  • L_node: 바쁜 큐빗 → 고품질 큐빗 (MLP)     │
│  τ annealing (1.0 → 0.05)                    │
└──────────────────────────────────────────────┘
```

- Non-differentiable barrier → PST 직접 최적화 불가
- Surrogate loss로 noise-aware 학습

**Speaker Notes:** Non-differentiable barrier 때문에 PST를 직접 최적화할 수 없어 미분 가능한 surrogate loss로 학습합니다. L_surr은 상호작용하는 큐빗 쌍을 error-weighted 최단경로 거리가 가까운 물리 큐빗에 배치하도록 유도하고, L_node는 MLP 기반 품질 점수로 바쁜 큐빗을 고품질 물리 큐빗에 유도합니다.

---

### Slide 7: Loss Design (상세)

**이 슬라이드는 수식과 다이어그램을 함께 사용. 각 loss의 직관을 시각적으로 표현해주세요.**

**Combined Loss:**
```
L = L_surr + α·L_node
```

**2개 loss를 각각 블록으로 구분하여 시각화:**

**(1) L_surr — Error-Aware Edge Quality Loss (Primary, 계수 1.0)**
- 논리적으로 상호작용하는 큐빗 쌍을 물리적으로 **error-weighted 최단경로 거리가 가까운** 위치에 매핑
- d_error(p,q) = cx_error weighted shortest path (Floyd-Warshall로 사전 계산)
- L_surr = (1/|E_circuit|) · Σ_{(i,j)∈E_circuit} Σ_{p,q} P_ip · P_jq · d_error(p,q)
- 단순 hop distance가 아닌 **error-weighted distance** → noise-aware 매핑
- **시각화:** 회로 그래프의 엣지 → 하드웨어 그래프의 error-weighted 최단경로 대응 그림

**(2) L_node — Node Quality Loss (α = 0.3)**
- 바쁜 논리 큐빗(2Q gate 많은) → 고품질 물리 큐빗에 매핑 유도
- q_score(p) = sigmoid(MLP(hw_features_p)) — **2-layer MLP** (7→16→1)
  - 7개 hardware noise features의 비선형 관계까지 포착
- L_node = (1/l) · (-Σ w(i) · Σ P_ip · q_score(p))
- **시각화:** 논리 큐빗(바쁜 정도) ↔ 물리 큐빗(품질 점수) 매칭 그림

**하단에 계수 표:**

| Loss | 계수 | 역할 |
|------|:----:|------|
| L_surr | 1.0 | Primary: error-weighted 거리 최소화 |
| L_node | α=0.3 | 노드 품질: 저오류 큐빗 선호 |

**Speaker Notes:** Surrogate loss 설계입니다. L_surr이 primary loss로, 상호작용하는 큐빗 쌍을 error-weighted 최단경로 거리가 가까운 물리 큐빗에 배치하도록 유도합니다. 단순 hop distance가 아닌 cx_error를 가중치로 한 최단경로를 사용하여 오류율이 낮은 경로를 선호합니다. Floyd-Warshall로 모든 쌍 간 거리를 사전 계산합니다. L_node는 MLP로 학습된 품질 점수를 사용하여 바쁜 논리 큐빗을 고품질 물리 큐빗에 유도합니다. 모든 loss term은 per-pair/per-qubit 정규화되어 회로/하드웨어 크기에 무관하게 스케일이 맞습니다.

---

### Slide 8: Dataset & Hardware Overview (1장으로 통합)

**좌우 또는 상하 분할:**

**상단: Dataset 요약 표**

| Dataset | 회로 수 | Label |
|---------|:-------:|:-----:|
| QUEKO | 540 labeled / 900 total | O |
| MLQD | 3,729 labeled / 4,443 total | O |
| MQT Bench | 1,219 | X |
| QASMBench | 94 | X |
| RevLib | 231 | X |
| **Total** | **6,887** (**4,269** labeled) | |
| Benchmark | 23 circuits (3Q~9Q, 평가 전용) | — |

- **Benchmark 회로는 훈련 데이터와 완전 분리** — 훈련 데이터셋에서 benchmark과 겹치는 회로는 사전에 제거 (deduplication)

**하단: Hardware 구성**
- **Training:** 49 Qiskit FakeBackendV2 (real noise만 사용, synthetic 제외)
  - QUEKO/MLQD 회로는 real backend에 재할당
- **Test (UNSEEN):** FakeToronto(27Q), FakeBrooklyn(65Q), FakeTorino(133Q)
  - "UNSEEN" 라벨을 빨간/주황으로 강조
- 의미: 학습 때 본 적 없는 하드웨어 → 진정한 hardware-agnostic 검증

**Speaker Notes:** 5개 데이터셋에서 총 6,887개 훈련 회로를 사용하고, 평가용 benchmark 23개 회로는 훈련 데이터와 완전히 분리했습니다. 훈련 데이터에서 benchmark과 겹치는 회로는 사전에 제거하여 공정한 평가를 보장합니다. 실제 noise를 가진 Qiskit FakeBackendV2만 사용하고, QUEKO/MLQD 회로는 real backend에 재할당합니다. 평가는 한 번도 본 적 없는 Toronto, Brooklyn, Torino 3개 백엔드에서 수행합니다.

---

### Slide 9: Evaluation Setup

**내용:**

**평가 프로토콜:**
- **Metric:** PST (Probability of Successful Trials) — 8,192 shots, Qiskit Aer noise simulation
- **Baselines (6가지):** SABRE, Trivial, Dense, NASSC, Noise-Adaptive, QAP(GraMA)
- **Benchmark:** 23개 standard circuits (3Q~9Q, 훈련 데이터와 완전 분리)
- **Test Backends (UNSEEN):** FakeToronto(27Q), FakeBrooklyn(65Q), FakeTorino(133Q)
- **공정 비교 조건:** 모든 layout method에 동일한 SABRE/NASSC routing + optimization_level 3 적용

**Speaker Notes:** 평가는 학습 때 한 번도 본 적 없는 3개 backend에서 수행합니다. 23개 benchmark 회로는 훈련 데이터와 완전 분리되어 있습니다. 모든 layout method에 동일한 SABRE routing과 Qiskit optimization level 3을 적용하여 공정하게 비교하고, PST를 8,192 shots noise simulation으로 측정합니다. Baseline은 SABRE, Trivial, Dense 같은 범용 heuristic부터 NASSC, Noise-Adaptive, GraMA 같은 noise-aware 방법까지 포함합니다.

---

### Slide 10: Experimental Results

**빈 페이지로 제작. 실험 결과 그래프/표를 직접 삽입할 예정.**
- 슬라이드 제목만 "Experimental Results"로 표시
- 나머지 영역은 비워둘 것

---

## 추가 제작 지시

1. **Figure 품질이 최우선:** Slide 3(전체 파이프라인), Slide 5(모델 아키텍처), Slide 6(학습 전략), Slide 7(Loss 설계)은 반드시 논문 figure 수준의 블록 다이어그램으로 제작해주세요. 텍스트 나열이 아니라 시각적 흐름도여야 합니다.
2. **색상 일관성:** 파이프라인 다이어그램의 색상 코딩을 전체 PPT에서 일관되게 유지 (예: Circuit=파랑, Hardware=초록, Cross-Attention=보라 등)
3. **Non-differentiable barrier:** Slide 2, 3, 6에서 이 개념을 시각적으로 명확히 표시 (점선, 색상 변경 등). 이 barrier가 전체 연구 설계의 핵심 동기.
4. **슬라이드 하단에 speaker notes 반드시 포함**
5. **폰트:** 본문 최소 18pt, 다이어그램 내 텍스트 최소 14pt. 읽기 어려운 작은 글씨 금지.
6. **슬라이드 번호:** 우하단에 표시
