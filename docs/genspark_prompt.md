# GenSpark PPT 생성 프롬프트

아래 프롬프트를 GenSpark에 입력하세요.

---

다음 지시에 따라 대학원 랩미팅용 PPT를 제작해주세요.

## 발표 상황
- **대상:** 지도교수님 (머신러닝/양자컴퓨팅에 익숙하시며, 큐빗 매핑 문제에 대한 기본 배경지식이 있으심)
- **성격:** 오랜만에 하는 연구 진행 보고 미팅 (처음이 아님). 배경 설명은 최소화하고 핵심 설계와 진행 상황에 집중
- **언어:** 한국어 기반, 기술 용어는 영어 그대로 (예: initial layout, cross-attention, surrogate loss, PST)
- **슬라이드 수:** 10장 내외 (컴팩트하게)
- **톤:** 학술적이되 랩미팅 수준으로 자연스럽게. 격식적인 학회 발표는 아님

## 디자인 & 시각화 핵심 요구사항

**가장 중요:** 이 PPT는 텍스트 위주가 아니라 **연구 논문의 figure 수준의 다이어그램/도식 위주**로 제작해주세요.
- 모델 아키텍처, 파이프라인, 학습 전략 등은 반드시 **시각적 다이어그램(figure)**으로 표현. 텍스트 나열 금지.
- 다이어그램은 학술 논문에 들어가는 것처럼 **블록 다이어그램, 화살표 흐름도, 색상 구분** 등을 활용하여 한눈에 구조가 파악되도록
- 색상 팔레트: 깔끔한 학술 스타일 (파스텔 또는 muted tone). 너무 화려하지 않게.
- 슬라이드당 텍스트는 키워드/짧은 문장 수준으로 최소화. 설명은 speaker notes에 작성.
- 각 슬라이드에 speaker notes를 반드시 포함해주세요.

---

## 슬라이드 구성 (10장)

### Slide 1: Title
**GraphQMap: Hardware-Agnostic Quantum Qubit Mapping via Graph Neural Networks**
- 이름, 소속, 날짜 (2024.03.24)
- 심플한 배경. 양자 회로나 그래프 네트워크를 연상시키는 추상적 이미지 가능.

---

### Slide 2: Problem & Motivation (1장으로 압축)

교수님이 이미 아시는 배경이므로 길게 설명하지 않고, **이 연구가 해결하는 문제**를 간결하게 상기시키는 수준.

**핵심 메시지:** 기존 initial layout 방법들은 모두 hardware-specific → 새 하드웨어마다 재설정 필요. 학습 기반 hardware-agnostic 접근이 없음.

**내용 (시각적으로 표현):**
- 왼쪽: 기존 방법 요약 표 (간략하게)

| 방법 | 핵심 한계 |
|------|-----------|
| SABRE | Noise 미고려 |
| Noise-Adaptive | 회로 구조 미고려 |
| NASSC | Hardware-specific 튜닝 |
| QAP | 확장성 부족 |
| OLSQ2 | 계산 비용 높음 |

- 오른쪽 또는 하단: **공통 한계 = Hardware-specific** → Research Gap 강조 (빨간 박스나 강조색)
- **화살표로 "→ Hardware-Agnostic 학습 기반 접근 필요"** 로 다음 슬라이드 연결

**Speaker Notes:** 기존 방법들은 모두 특정 하드웨어에 종속적입니다. 새 하드웨어마다 재설정이 필요하고, 하나의 모델이 다양한 하드웨어에서 범용 동작하는 학습 기반 접근은 아직 없습니다.

---

### Slide 3: Research Objective

**핵심 메시지:** 단일 GNN 모델 → 임의의 양자 하드웨어에서 고품질 initial layout 출력

**시각적 구성:**
- 중앙에 핵심 목표를 큰 텍스트: "One model, any hardware"
- 4가지 설계 목표를 2×2 격자로 아이콘과 함께:
  1. **Hardware-agnostic:** fine-tuning 없이 unseen 하드웨어에서도 동작
  2. **Noise-aware:** 하드웨어 noise 특성을 피처로 활용
  3. **Fast inference:** solver 대비 실시간 추론
  4. **Modular:** initial layout만 담당, routing은 SABRE에 위임
- 하단: Primary Metric = **PST (Probability of Successful Trials)**

**Speaker Notes:** GraphQMap의 목표는 하나의 모델이 60개 백엔드(5Q~127Q)로 학습하고, 한 번도 본 적 없는 3개 백엔드(Toronto 27Q, Brooklyn 65Q, Torino 133Q)에서 평가하는 것입니다. 모델은 initial layout만 출력하고, routing은 Qiskit SABRE에 맡깁니다.

---

### Slide 4: Overall Pipeline (핵심 Figure)

**이 슬라이드가 PPT에서 가장 중요한 figure입니다. 논문 Figure 1 수준으로 완성도 높게 제작해주세요.**

**다이어그램 구성 (위에서 아래로 흐르는 블록 다이어그램):**

```
[입력층 - 좌우 병렬]
  Quantum Circuit (.qasm)              Target Hardware (Backend)
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

### Slide 5: Graph Representation (Circuit + Hardware를 1장에)

**좌우 분할 레이아웃으로 Circuit Graph / Hardware Graph를 한 슬라이드에:**

**왼쪽: Circuit Graph**
- 간단한 양자 회로 다이어그램(4큐빗) → 화살표 → 변환된 그래프(4노드)
- Node Features: gate_count, 2Q_gate_count, degree, depth_participation
- Edge Features: interaction_count, earliest/latest_interaction
- 정규화: 회로 내 z-score (상대적 중요도)

**오른쪽: Hardware Graph**
- 하드웨어 토폴로지(예: heavy-hex 일부) → 노드 색상으로 품질 표현(진한=고품질)
- Node Features: T1, T2, readout_error, sq_error, degree, t1_cx_ratio, t2_cx_ratio
- Edge Features: cx_error
- 정규화: **백엔드 내 z-score** → 이것이 hardware-agnostic의 핵심 메커니즘임을 강조 (별도 박스나 색상으로)

**Speaker Notes:** 회로는 큐빗=노드, 2Q gate=엣지로 변환하고, 큐빗의 "바쁜 정도"를 feature로 인코딩합니다. 하드웨어는 물리 큐빗=노드, coupling map=엣지로 변환하고, noise 특성을 feature로 인코딩합니다. 핵심 설계는 각각 회로 내/백엔드 내에서 z-score 정규화하는 것입니다. 절대값 대신 상대적 순위를 사용하여 5Q 칩과 127Q 칩을 동일한 feature 공간에서 비교 가능하게 합니다. 이것이 hardware-agnostic 일반화의 핵심입니다.

---

### Slide 6: Model Architecture Detail (Dual GNN + Cross-Attention + Score Head)

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

**(3) Score Head → Sinkhorn → Hungarian (하단)**
- Score: S_ij = scaled dot-product of projected C', H'
- 3단계 변환을 **히트맵 시각화**로: Score Matrix(수치) → Sinkhorn(부드러운 분포) → Hungarian(이진 할당)
- τ가 soft↔hard를 제어한다는 것을 작은 보조 그림으로

**Speaker Notes:** GATv2 기반 3-layer GNN이 각각 회로와 하드웨어를 인코딩합니다. Cross-Attention이 이 모델의 핵심인데, 단순 dot-product는 유사한 것끼리 매칭하지만, 실제로는 바쁜 논리 큐빗을 고품질 물리 큐빗에 배치하는 상보적 관계가 필요합니다. 양방향 cross-attention으로 이를 포착합니다. Score Head에서 모든 쌍의 적합도를 계산하고, Sinkhorn으로 soft assignment, 추론 시 Hungarian으로 이산 매핑을 결정합니다.

---

### Slide 7: Training Strategy — 2-Stage Curriculum

**다이어그램 위주. 텍스트 최소화.**

**시각적 구성 (위→아래 흐름도):**

```
┌──────────────────────────────────────────────┐
│  Stage 1: Supervised Pre-training            │
│                                              │
│  Phase 1: MLQD + QUEKO (3,846 labeled)       │
│  • CE Loss (Sinkhorn output vs ground truth) │
│  • τ annealing: 1.0 → 0.05                  │
│           ↓                                  │
│  Phase 2: QUEKO only (486 circuits)          │
│  • CE Loss, LR ×1/10                        │
│  • Router-agnostic optimal로 보정            │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│  Stage 2: Unsupervised Surrogate Fine-tuning │
│                                              │
│  전체 6,887 회로 (label 불필요)               │
│  L = L_surr + α·L_node + λ·L_sep            │
│  • L_surr: error-weighted 최단경로 거리 최소화│
│  • L_node: 바쁜 큐빗 → 고품질 큐빗 유도      │
│  • L_sep: 회로간 물리적 분리 (향후 확장용)    │
│  Fixed τ = 0.05                              │
└──────────────────────────────────────────────┘
```

**오른쪽에 보조 설명 박스: "왜 2-Stage인가?"**
- Non-differentiable barrier → PST 직접 최적화 불가
- Stage 1: label로 "좋은 매핑의 기본 구조" 학습 (cold start 해결)
- Stage 2: surrogate loss로 noise-aware fine-tuning

**Speaker Notes:** Non-differentiable barrier 때문에 PST를 직접 최적화할 수 없어 2단계 전략을 사용합니다. Stage 1에서 기존 label(MLQD의 OLSQ2 label + QUEKO의 optimal label)로 기본 매핑 감각을 학습합니다. Phase 2에서 QUEKO의 router-agnostic 정답으로 보정합니다. Stage 2에서는 label 없이 계산 가능한 surrogate loss로 noise를 고려한 fine-tuning을 합니다. L_surr은 error-weighted 최단경로 거리를, L_node는 learnable 품질 점수를 사용합니다.

---

### Slide 8: Dataset & Hardware Overview (1장으로 통합)

**좌우 또는 상하 분할:**

**상단: Dataset 요약 표**

| Dataset | 회로 수 | Label | Stage | Label Source |
|---------|:-------:|:-----:|-------|-------------|
| QUEKO | 540 labeled / 900 total | O | 1+2 | τ⁻¹ optimal (zero-SWAP) |
| MLQD | 3,729 labeled / 4,443 total | O | 1+2 | OLSQ2 solver 역추적 |
| MQT Bench | 1,219 | X | 2 | — |
| QASMBench | 94 | X | 2 | — |
| RevLib | 231 | X | 2 | — |
| **Total** | **6,887** | **4,269** | | |

**하단: Hardware 구성**
- Training: 60 backends (55 Qiskit FakeBackendV2 + 5 synthetic), 5Q~127Q
- **Test (UNSEEN):** FakeToronto(27Q), FakeBrooklyn(65Q), FakeTorino(133Q)
  - "UNSEEN" 라벨을 빨간/주황으로 강조
- 의미: 학습 때 본 적 없는 하드웨어 → 진정한 hardware-agnostic 검증

**Speaker Notes:** 5개 데이터셋에서 총 6,887개 회로를 사용합니다. 4,269개가 label을 갖고 있어 Stage 1에, 전체는 Stage 2에 사용됩니다. 하드웨어는 60개로 학습하고, 한 번도 본 적 없는 Toronto, Brooklyn, Torino 3개 백엔드에서 평가하여 hardware-agnostic 성능을 검증합니다.

---

### Slide 9: Current Progress & Evaluation Setup

**내용:**

**구현 완료 (체크리스트 형태):**
- ✓ 데이터 파이프라인 (5개 데이터셋 수집/전처리/split)
- ✓ 전체 모델 아키텍처 (Dual GNN, Cross-Attention, Score Head, Sinkhorn, Hungarian)
- ✓ 2-Stage 학습 루프 (CE loss, surrogate losses, τ annealing, early stopping)
- ✓ 평가 프레임워크 (PST 측정, 6가지 baseline 비교, benchmark runner)
- ✓ 119개 unit/integration tests

**평가 프로토콜 (간단히):**
- Metric: PST (8,192 shots, noise simulation)
- Baselines: SABRE, Trivial, Dense, NASSC, Noise-Adaptive, QAP
- Benchmark: 8개 standard circuits (3Q~9Q)
- 현재: 학습 실험 진행 중

**Speaker Notes:** 전체 시스템이 구현 완료되었고 119개 테스트로 검증했습니다. PST를 noise simulation으로 측정하고 6가지 baseline과 비교합니다. 현재 학습 실험을 진행하고 있습니다.

---

### Slide 10: Expected Contributions & Next Steps

**왼쪽: Contributions (번호 리스트)**
1. **Hardware-Agnostic Mapping:** 단일 모델 → unseen 하드웨어에서 fine-tuning 없이 동작
2. **Noise-Aware Surrogate Loss:** non-differentiable barrier를 우회하는 미분 가능 loss 설계
3. **Cross-Attention Architecture:** 유사성이 아닌 상보성 포착
4. **Learnable Quality Score:** 학습된 w1~w5로 noise factor 중요도 해석 가능

**오른쪽: Next Steps (타임라인 또는 리스트)**
- Stage 1, 2 학습 완료 + 하이퍼파라미터 튜닝
- Unseen backend PST 평가
- Ablation study (Cross-Attention, L_node, τ 전략 등)
- 논문 작성

**Speaker Notes:** 주요 기여는 hardware-agnostic 학습 기반 매핑, noise-aware surrogate loss 설계, cross-attention 기반 상보적 매칭, 그리고 해석 가능한 noise factor 분석입니다. 향후 학습 완료 후 unseen backend에서 PST 평가와 ablation study를 진행하고 논문을 작성할 예정입니다.

---

## 추가 제작 지시

1. **Figure 품질이 최우선:** Slide 4(전체 파이프라인), Slide 6(모델 아키텍처), Slide 7(학습 전략)은 반드시 논문 figure 수준의 블록 다이어그램으로 제작해주세요. 텍스트 나열이 아니라 시각적 흐름도여야 합니다.
2. **색상 일관성:** 파이프라인 다이어그램의 색상 코딩을 전체 PPT에서 일관되게 유지 (예: Circuit=파랑, Hardware=초록, Cross-Attention=보라 등)
3. **Non-differentiable barrier:** Slide 4와 7에서 이 개념을 시각적으로 명확히 표시 (점선, 색상 변경 등). 이 barrier가 전체 연구 설계의 핵심 동기.
4. **슬라이드 하단에 speaker notes 반드시 포함**
5. **폰트:** 본문 최소 18pt, 다이어그램 내 텍스트 최소 14pt. 읽기 어려운 작은 글씨 금지.
6. **슬라이드 번호:** 우하단에 표시
