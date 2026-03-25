# GraphQMap PPT 제작 설명서

> 이 문서는 PPT 제작 AI에게 전달하기 위한 설명서입니다.
> 발표 대상은 대학원 지도교수님이며, 프로젝트에 대한 첫 번째 미팅입니다.
> 교수님은 머신러닝/양자컴퓨팅 전반에 익숙하시지만, 이 프로젝트의 구체적인 설계 결정은 모르시므로 **왜 이렇게 설계했는지** 논리적 흐름이 중요합니다.

---

## 발표 톤 & 스타일

- 학술 발표 톤 (학회 발표가 아닌 랩미팅 수준, 지나치게 격식적이지 않게)
- 한국어 기반, 기술 용어는 영어 그대로 사용 (예: "initial layout", "cross-attention", "surrogate loss")
- 슬라이드당 텍스트는 최소화하고 다이어그램/도식 위주로
- 각 슬라이드 하단에 발표자 노트(speaker notes)를 별도 작성해주세요

---

## 용어 사전 (PPT 제작 시 참고)

PPT 내에 용어 사전 슬라이드를 만들 필요는 없습니다. 제작 시 문맥을 이해하기 위한 참고용입니다.

| 용어 | 설명 |
|------|------|
| **Qubit (큐빗)** | 양자 컴퓨터의 기본 연산 단위. 고전적 bit의 양자 버전. |
| **Logical Qubit (논리 큐빗)** | 양자 회로(프로그램)에서 사용하는 추상적인 큐빗. 프로그래머가 코드에서 다루는 변수와 유사. |
| **Physical Qubit (물리 큐빗)** | 실제 양자 하드웨어 칩에 존재하는 물리적 큐빗. 각각 성능(noise)이 다르고, 특정 이웃 큐빗과만 연결됨. |
| **Quantum Circuit (양자 회로)** | 양자 프로그램. 큐빗에 게이트(연산)를 순서대로 적용하는 구조. 고전적 프로그램의 회로도에 해당. |
| **Gate (게이트)** | 큐빗에 적용하는 연산. 1-qubit gate(단일 큐빗 회전)와 2-qubit gate(두 큐빗 간 상호작용, 예: CNOT/CX)가 있음. |
| **CNOT / CX** | 가장 기본적인 2-qubit gate. 두 큐빗이 물리적으로 연결되어 있어야 실행 가능. |
| **SWAP gate** | 두 큐빗의 상태를 교환하는 게이트. 물리적으로 연결되지 않은 큐빗 간 2-qubit gate를 실행하기 위해 삽입됨. CNOT 3개로 분해되므로 비용이 큼. |
| **NISQ** | Noisy Intermediate-Scale Quantum. 현재 세대의 양자 컴퓨터를 지칭. 큐빗 수가 제한적(수십~수백 개)이고 noise가 큼. |
| **Noise** | 양자 하드웨어의 불완전성으로 인한 오류. 게이트 실행 오류, 큐빗 상태 붕괴(T1/T2), 측정 오류 등. noise가 클수록 계산 결과가 부정확해짐. |
| **T1 (에너지 완화 시간)** | 큐빗이 excited state에서 ground state로 떨어지는 데 걸리는 시간. 길수록 좋음. |
| **T2 (위상 완화 시간)** | 큐빗의 양자 정보(위상)가 유지되는 시간. 길수록 좋음. |
| **Coupling Map** | 하드웨어에서 어떤 물리 큐빗 쌍이 직접 연결되어 있는지를 나타내는 그래프. 연결되지 않은 쌍은 SWAP 없이 2-qubit gate 실행 불가. |
| **Quantum Circuit Compilation** | 추상적 양자 회로를 특정 하드웨어에서 실행 가능한 형태로 변환하는 과정. Initial Layout → Routing → Optimization → Scheduling 단계로 구성. |
| **Initial Layout (초기 배치)** | 컴파일의 첫 단계. 각 논리 큐빗을 어떤 물리 큐빗에 배치할지 결정. 이 선택이 이후 모든 단계의 품질을 좌우함. 이 프로젝트의 핵심 출력. |
| **Routing** | Initial layout 이후, 물리적으로 연결되지 않은 큐빗 간 게이트를 실행하기 위해 SWAP gate를 삽입하는 단계. |
| **SABRE** | Qiskit의 기본 routing 알고리즘. Heuristic 기반으로 SWAP 횟수를 최소화. 본 프로젝트에서 routing 담당으로 사용. |
| **Transpile (트랜스파일)** | 양자 회로 컴파일의 Qiskit 용어. Layout + Routing + Optimization을 모두 포함하는 변환 과정. |
| **PST (Probability of Successful Trials)** | 양자 회로 실행 결과가 정답과 일치할 확률. 본 프로젝트의 primary metric. 높을수록 좋음. noise가 적을수록 높아짐. |
| **FakeBackendV2** | Qiskit에서 제공하는 실제 하드웨어의 시뮬레이션 모델. 실제 칩의 topology와 noise 특성을 그대로 복제. 실물 하드웨어 없이 실험 가능. |
| **GNN (Graph Neural Network)** | 그래프 구조 데이터를 처리하는 신경망. 노드와 엣지의 feature를 이웃 정보와 함께 학습. 양자 회로와 하드웨어 모두 그래프로 자연스럽게 표현됨. |
| **GATv2** | Graph Attention Network v2. GNN의 한 종류로, 이웃 노드에 대한 attention weight를 동적으로 계산하여 중요한 이웃에 더 집중. |
| **Cross-Attention** | 두 개의 서로 다른 시퀀스(여기서는 회로 임베딩과 하드웨어 임베딩) 간의 attention. Transformer의 encoder-decoder attention과 동일한 메커니즘. |
| **Sinkhorn Algorithm** | 행렬을 doubly stochastic matrix(각 행과 열의 합이 1)로 변환하는 반복 알고리즘. 연속적(미분 가능)인 soft assignment를 생성. |
| **Hungarian Algorithm** | 이분 매칭(bipartite matching) 문제의 최적해를 구하는 알고리즘. Sinkhorn의 soft assignment를 discrete한 1:1 매핑으로 변환. 추론 시에만 사용. |
| **Doubly Stochastic Matrix** | 모든 행의 합과 모든 열의 합이 각각 1인 비음수 행렬. 1:1 매핑의 연속적 완화(relaxation). |
| **Temperature (tau, τ)** | Sinkhorn의 sharpness를 제어하는 파라미터. 높으면 uniform에 가까운 soft assignment, 낮으면 one-hot에 가까운 hard assignment. 학습 중 점차 낮춤(annealing). |
| **Non-differentiable Barrier** | 모델 출력(layout) → Qiskit transpile → PST 사이에 미분이 불가능한 구간. transpile이 미분 불가능하므로 PST에서 모델로 직접 gradient를 전달할 수 없음. 이것이 2-stage 학습 전략의 근본 원인. |
| **Surrogate Loss** | 직접 최적화할 수 없는 목표(PST)를 대신하여 설계한 미분 가능한 대리 손실 함수. PST와 상관관계가 높도록 설계. |
| **Cross-Entropy Loss** | 분류 문제의 표준 손실 함수. 여기서는 모델이 출력한 soft assignment(P)와 정답 매핑(permutation matrix) 간의 차이를 측정. |
| **SMT Solver** | Satisfiability Modulo Theories. 논리적 제약 조건을 만족하는 해를 찾는 수학적 도구. OLSQ2가 이를 사용하여 최적 매핑을 계산. |
| **OLSQ2** | Optimal Layout Synthesis for Quantum computing. SMT solver 기반으로 layout과 routing을 동시에 최적화하는 도구. 최적에 가깝지만 계산 비용이 매우 높음. |
| **QUEKO** | Quantum Mapping Examples with Known Optimal. SWAP 0개로 매핑 가능하도록 역설계된 벤치마크 회로 세트. 정답 layout이 수학적으로 보장됨. |
| **MLQD** | Machine Learning for Quantum Design. OLSQ2로 최적화된 회로를 포함하는 데이터셋. 결과 회로에서 SWAP 패턴을 역추적하여 initial layout label을 추출. |
| **Floyd-Warshall** | 그래프의 모든 노드 쌍 간 최단 경로를 계산하는 알고리즘. 하드웨어 그래프에서 모든 물리 큐빗 쌍 간의 hop distance를 사전 계산하는 데 사용. |
| **Adjacency (인접)** | 두 물리 큐빗이 coupling map에서 직접 연결되어 있는 상태 (hop distance = 1). 인접한 큐빗 쌍은 SWAP 없이 2-qubit gate 실행 가능. |
| **Hop Distance** | 하드웨어 그래프에서 두 물리 큐빗 간 최소 이동 횟수. hop distance 1 = 인접(SWAP 불필요), 2 이상 = SWAP 필요. |

---

## 슬라이드별 상세 설명

### Slide 1: Title

**제목:** GraphQMap: Hardware-Agnostic Quantum Qubit Mapping via Graph Neural Networks

**내용:**
- 제목, 발표자 이름, 소속, 날짜

**디자인:** 깔끔한 타이틀 슬라이드. 양자 회로나 그래프 네트워크를 연상시키는 추상적 배경 이미지 사용 가능.

---

### Slide 2: Background — Quantum Circuit Compilation

**이 슬라이드의 목적:** 교수님에게 "왜 이 문제가 중요한가"의 출발점을 제시. 양자 회로가 실제 하드웨어에서 실행되려면 컴파일이 필요하다는 점을 설명.

**핵심 메시지:** 양자 회로(프로그램)를 실제 하드웨어에서 실행하려면 "컴파일" 과정이 필요하며, 이 과정에서 품질 손실이 발생한다.

**내용:**
- 양자 회로는 논리 큐빗(logical qubit)으로 작성됨
- 실제 NISQ 하드웨어에서 실행하려면 물리 큐빗(physical qubit)에 매핑해야 함
- 하드웨어 제약 2가지:
  - **제한된 연결성(limited connectivity):** 모든 큐빗 쌍이 직접 연결되어 있지 않음. 연결되지 않은 큐빗 간 2-qubit gate를 실행하려면 SWAP gate를 삽입해야 함
  - **불균일한 noise:** 큐빗마다, 연결마다 오류율이 다름 (T1, T2, gate error, readout error)
- SWAP gate 삽입 → 회로 깊이 증가 → noise 누적 → 실행 결과 품질 저하

**다이어그램 제안:**
- 왼쪽: 4개 논리 큐빗이 자유롭게 연결된 양자 회로 (all-to-all 연결)
- 오른쪽: 물리 큐빗이 격자(grid) 형태로 제한적으로 연결된 하드웨어
- 화살표로 "매핑 필요" 표시
- 연결되지 않은 큐빗 간 게이트 실행 시 SWAP 삽입이 필요함을 시각적으로 표현

**Speaker Notes:**
양자 회로는 이상적인 환경을 가정하고 작성되지만, 실제 양자 칩은 큐빗 간 연결이 제한적이고 각 큐빗의 품질도 다릅니다. 따라서 논리 큐빗을 물리 큐빗에 매핑하는 "컴파일" 과정이 반드시 필요하며, 이 과정에서 추가되는 SWAP gate가 회로의 noise를 크게 증가시킵니다.

---

### Slide 3: Background — Initial Layout의 중요성

**이 슬라이드의 목적:** 컴파일 단계 중 "Initial Layout"이 가장 결정적이라는 점을 강조. 프로젝트가 왜 이 특정 단계에 집중하는지 정당화.

**핵심 메시지:** 같은 회로, 같은 하드웨어에서도 초기 배치(layout)에 따라 결과 품질이 극적으로 달라진다.

**내용:**
- Quantum Circuit Compilation의 4단계: **Initial Layout** → Routing → Optimization → Scheduling
- Initial Layout = 각 논리 큐빗을 어떤 물리 큐빗에 배치할지 결정하는 첫 단계
- Layout 품질의 영향:
  - 좋은 layout → 필요한 SWAP 수 최소 → 회로 깊이 짧음 → PST(성공 확률) 높음
  - 나쁜 layout → SWAP 폭증 → noise 누적 → 실행 결과 무의미
- **핵심 포인트:** 같은 회로 + 같은 하드웨어에서도 layout 선택에 따라 PST가 크게 달라짐

**다이어그램 제안:**
- 컴파일 파이프라인을 4단계 화살표로 표현: Initial Layout → Routing → Optimization → Scheduling
- Initial Layout 단계를 강조(색상, 크기)
- 아래에 "Good Layout vs Bad Layout" 비교:
  - Good: SWAP 2개 삽입 → 짧은 회로 → PST 높음
  - Bad: SWAP 10개 삽입 → 긴 회로 → PST 낮음

**Speaker Notes:**
컴파일의 첫 단계인 Initial Layout이 전체 품질을 결정합니다. Routing 알고리즘이 아무리 좋아도, 초기 배치가 나쁘면 많은 SWAP gate가 필요해지고, 이는 회로 깊이와 noise를 크게 증가시킵니다. 반대로 좋은 초기 배치는 SWAP을 최소화하여 높은 PST를 달성할 수 있습니다. 이것이 저희가 Initial Layout에 집중하는 이유입니다.

---

### Slide 4: Background — 기존 방법의 한계

**이 슬라이드의 목적:** 기존 접근법들을 정리하고, 공통 한계를 도출하여 research gap을 명확히 함.

**핵심 메시지:** 기존 방법들은 모두 **특정 하드웨어에 종속**되며, 하나의 모델이 다양한 하드웨어에 범용적으로 동작하는 학습 기반 접근은 부재.

**내용 (표):**

| 방법 | 접근 방식 | 한계 |
|------|-----------|------|
| SABRE | Heuristic: SWAP 횟수 최소화 | Noise 특성을 고려하지 않음 |
| Noise-Adaptive | Noise 기반 subgraph 매칭 | 회로의 구조(어떤 큐빗이 바쁜지)를 고려하지 않음 |
| NASSC | Noise + SWAP 비용 통합 | 하드웨어별 파라미터 튜닝 필요 |
| QAP | Quadratic Assignment 최적화 | 큐빗 수 증가 시 계산량 폭증 (확장성 부족) |
| OLSQ2 | SMT Solver로 layout+routing 동시 최적화 | 수학적 최적에 가깝지만 계산 비용 매우 높아 실시간 사용 불가 |

- 공통 한계: **특정 하드웨어에 종속** → 새로운 하드웨어가 나올 때마다 재설정/재계산 필요
- **Research Gap:** 하나의 학습된 모델이 다양한 하드웨어에서 fine-tuning 없이 범용 동작하는 접근이 없음

**디자인 제안:**
- 표 형태로 정리
- 하단에 "공통 한계"를 빨간색 등 강조색으로 별도 박스에 표시
- Research Gap을 화살표로 연결하여 다음 슬라이드(Research Objective)로 자연스럽게 이어지게

**Speaker Notes:**
기존 방법들을 크게 다섯 가지로 분류할 수 있습니다. SABRE는 빠르지만 noise를 무시하고, OLSQ2는 최적에 가깝지만 계산 비용이 너무 높습니다. 중요한 것은, 이 모든 방법이 특정 하드웨어에 종속적이라는 점입니다. 새로운 양자 칩이 출시될 때마다 처음부터 다시 설정하거나 계산해야 합니다. 저희는 이 gap을 학습 기반 접근으로 해결하고자 합니다.

---

### Slide 5: Research Objective

**이 슬라이드의 목적:** 프로젝트의 목표를 명확하고 간결하게 선언.

**핵심 메시지:** "하나의 GNN 모델이 다양한 양자 하드웨어에서 high-quality initial layout을 출력한다."

**내용:**
- **핵심 목표:** 단일 GNN 모델 → 다양한 양자 하드웨어에서 고품질 initial layout 출력
- 4가지 설계 목표:
  1. **Hardware-agnostic:** 하드웨어별 fine-tuning 없이 동작 (학습 때 본 적 없는 하드웨어에서도)
  2. **Noise-aware:** 하드웨어 noise 특성을 입력 피처로 활용하여 고품질 큐빗에 우선 배치
  3. **Fast inference:** SMT solver(분~시간) 대비 실시간 추론(밀리초) 가능
  4. **Modular:** 모델은 initial layout만 담당, routing 이후는 기존 도구(Qiskit SABRE)에 위임
- **Primary Metric:** PST (Probability of Successful Trials) — 양자 회로 실행 결과가 정답과 일치할 확률

**디자인 제안:**
- 중앙에 핵심 목표를 큰 텍스트로
- 4가지 설계 목표를 아이콘과 함께 2x2 격자로 배치
- PST 메트릭을 하단에 별도 강조

**Speaker Notes:**
저희 모델 GraphQMap의 목표는 하나의 모델이 어떤 양자 하드웨어에든 범용적으로 동작하는 것입니다. 60개의 다양한 백엔드(5큐빗~127큐빗)로 학습하고, 학습 시 한 번도 본 적 없는 3개 백엔드에서 평가합니다. 모델은 initial layout만 출력하고, 이후 routing은 Qiskit의 SABRE 알고리즘에 맡깁니다.

---

### Slide 6: Core Pipeline Overview

**이 슬라이드의 목적:** 전체 시스템의 흐름을 한눈에 보여주는 핵심 다이어그램. 이후 슬라이드들이 각 블록을 상세 설명하므로, 여기서는 전체 그림만 제시.

**핵심 메시지:** 회로와 하드웨어를 각각 그래프로 변환 → 별도 GNN으로 인코딩 → Cross-Attention으로 상호 참조 → Score 계산 → Sinkhorn/Hungarian으로 매핑 결정.

**다이어그램 (이 슬라이드가 가장 중요한 다이어그램):**

```
[입력]
  양자 회로 (.qasm)          타겟 하드웨어 (FakeBackendV2)
       |                              |
       v                              v
[그래프 구성]
  Circuit Graph                Hardware Graph
  (노드=논리큐빗,               (노드=물리큐빗,
   엣지=2Q gate 연결)            엣지=coupling map)
       |                              |
       v                              v
[GNN 인코딩]
  Circuit GNN Encoder          Hardware GNN Encoder
  (GATv2 x 3 layers)          (GATv2 x 3 layers)
       |                              |
       v                              v
     C (l x d)                     H (h x d)
       |                              |
       +----------+-------------------+
                  |
                  v
[Cross-Attention Module]  (2 layers, bidirectional)
                  |
                  v
           C' (l x d), H' (h x d)
                  |
                  v
[Score Head]  →  S (l x h)  (각 논리-물리 큐빗 쌍의 매핑 적합도)
                  |
                  v
[Dummy Padding]  →  S (h x h)  (정방 행렬로 확장)
                  |
                  v
[Log-domain Sinkhorn]  →  P (h x h)  (doubly stochastic soft assignment)
                  |
                  v
[Hungarian Algorithm]  →  이산적 1:1 매핑 (추론 시만)
                  |
                  v
[출력: Initial Layout]  (논리 큐빗 i → 물리 큐빗 p_i)
                  |
                  v
- - - - - - - - NON-DIFFERENTIABLE BARRIER - - - - - - - -
                  |
                  v
[Qiskit Transpile + SABRE Routing]  →  컴파일된 회로
                  |
                  v
[Noise Simulation]  →  PST 측정
```

- **중요:** "NON-DIFFERENTIABLE BARRIER" 부분을 시각적으로 명확히 표시 (점선, 색상 변경 등). 이 barrier가 전체 학습 전략 설계의 핵심 동기임.
- l = 논리 큐빗 수, h = 물리 큐빗 수, d = embedding dimension (64)
- l은 항상 h보다 작음 (논리 큐빗 수 < 물리 큐빗 수)

**Speaker Notes:**
전체 파이프라인입니다. 양자 회로와 하드웨어를 각각 그래프로 변환하고, 별도의 GNN으로 인코딩합니다. Cross-Attention으로 두 임베딩이 서로를 참조한 뒤, Score Head에서 각 논리-물리 큐빗 쌍의 매핑 적합도를 계산합니다. Sinkhorn으로 soft assignment를 만들고, 추론 시에는 Hungarian 알고리즘으로 이산적 1:1 매핑을 결정합니다. 핵심은 중간의 non-differentiable barrier입니다. Qiskit transpile을 통과해야 최종 PST를 알 수 있는데, transpile은 미분 불가능하므로 PST에서 모델로 직접 gradient를 전달할 수 없습니다. 이 제약이 저희 2단계 학습 전략의 근본 원인입니다.

---

### Slide 7: Graph Representation — Circuit Graph

**이 슬라이드의 목적:** 양자 회로를 GNN 입력으로 어떻게 변환하는지 설명.

**핵심 메시지:** 양자 회로의 각 큐빗을 노드로, 2-qubit gate 연결을 엣지로 표현하고, 각 큐빗/연결의 "바쁜 정도"를 feature로 인코딩.

**내용:**
- **노드:** 각 논리 큐빗 = 1개 노드
- **엣지:** 2-qubit gate로 연결된 큐빗 쌍마다 1개 엣지 (같은 쌍의 여러 gate는 하나의 엣지로 merge, undirected)

- **Node Features** (각 큐빗이 "얼마나 바쁘고 중요한가"):

| Feature | 설명 |
|---------|------|
| gate_count | 해당 큐빗에 적용되는 총 게이트 수 |
| two_qubit_gate_count | 해당 큐빗이 참여하는 2-qubit gate 수 |
| degree | 이 큐빗과 2-qubit gate로 상호작용하는 다른 큐빗 수 |
| circuit_depth_participation | 전체 회로 depth 중 이 큐빗이 활성인 비율 (0~1) |

- **Edge Features** (두 큐빗 간 상호작용의 "강도와 시점"):

| Feature | 설명 |
|---------|------|
| interaction_count | 이 큐빗 쌍 간 2-qubit gate 수 |
| earliest_interaction | 첫 상호작용의 normalized 시점 (0~1) |
| latest_interaction | 마지막 상호작용의 normalized 시점 (0~1) |

- **정규화:** 각 feature를 해당 회로 내에서 z-score 정규화. "이 큐빗이 이 회로에서 상대적으로 얼마나 바쁜가"를 표현. (절대적 수치가 아닌 상대적 비율)

**다이어그램 제안:**
- 왼쪽: 간단한 양자 회로 다이어그램 (4 큐빗, 여러 gate)
- 오른쪽: 변환된 그래프 (4개 노드, 엣지에 feature 표시)
- 화살표로 변환 과정 표현

**Speaker Notes:**
양자 회로를 그래프로 변환합니다. 각 논리 큐빗이 노드가 되고, 2-qubit gate로 연결된 큐빗 쌍이 엣지가 됩니다. 노드 feature는 해당 큐빗이 회로에서 얼마나 바쁜지를 나타내고, 엣지 feature는 두 큐빗 간 상호작용의 강도와 시점을 인코딩합니다. 중요한 점은, feature를 각 회로 내에서 z-score 정규화한다는 것입니다. 이를 통해 회로의 절대적 크기에 상관없이 "이 큐빗이 이 회로 내에서 상대적으로 얼마나 중요한가"를 표현합니다.

---

### Slide 8: Graph Representation — Hardware Graph

**이 슬라이드의 목적:** 하드웨어를 GNN 입력으로 어떻게 변환하는지 설명. 특히 noise feature의 역할 강조.

**핵심 메시지:** 하드웨어의 topology(연결 구조)와 noise 특성(각 큐빗/연결의 품질)을 그래프 feature로 인코딩. 각 백엔드 내에서 정규화하여 cross-hardware 일반화를 가능하게 함.

**내용:**
- **노드:** 각 물리 큐빗 = 1개 노드
- **엣지:** Coupling map의 연결 (undirected)

- **Node Features** (각 물리 큐빗의 "품질"):

| Feature | 설명 | 의미 |
|---------|------|------|
| T1 | 에너지 완화 시간 | 길수록 고품질 |
| T2 | 위상 완화 시간 | 길수록 고품질 |
| frequency | 큐빗 주파수 | 구조적 정보 |
| readout_error | 측정 오류율 | 낮을수록 고품질 |
| single_qubit_error | 1-qubit gate 오류율 | 낮을수록 고품질 |
| degree | connectivity 차수 (이웃 수) | 연결이 많을수록 routing 유리 |

- **Edge Features** (각 연결의 "품질"):

| Feature | 설명 |
|---------|------|
| cx_error | 2-qubit gate 실행 시 오류율 |
| cx_duration | 2-qubit gate 실행 시간 |

- **정규화: 각 백엔드 내에서 z-score 정규화**
  - 핵심 설계: "이 큐빗이 이 칩 안에서 상대적으로 얼마나 좋은가"를 표현
  - 절대값이 아닌 상대적 순위를 학습 → 서로 다른 칩(5큐빗 vs 127큐빗, 구세대 vs 신세대)에서도 동일한 feature 공간에서 비교 가능
  - 이것이 hardware-agnostic 일반화의 핵심 메커니즘

**다이어그램 제안:**
- 실제 하드웨어 topology 예시 (예: IBM 27-qubit heavy-hex 격자)
- 노드 색상으로 품질 표현 (진한 색 = 고품질, 연한 색 = 저품질)
- 엣지 굵기로 연결 품질 표현 (굵은 = 저오류, 얇은 = 고오류)

**Speaker Notes:**
하드웨어도 그래프로 표현합니다. 각 물리 큐빗이 노드, coupling map의 연결이 엣지입니다. 중요한 것은 noise feature입니다. T1, T2, gate error 등을 통해 각 큐빗과 연결의 품질을 모델에 알려줍니다. 핵심 설계 결정은 **각 백엔드 내에서 z-score 정규화**하는 것입니다. 5큐빗 칩과 127큐빗 칩은 절대적 noise 수치가 완전히 다르지만, "이 큐빗이 이 칩 안에서 상대적으로 좋은 편인가?"라는 상대적 정보는 범용적입니다. 이것이 하나의 모델이 다양한 하드웨어에서 동작할 수 있는 핵심 메커니즘입니다.

---

### Slide 9: Model Architecture — Dual GNN Encoder

**이 슬라이드의 목적:** GNN 인코더의 구조 설명.

**핵심 메시지:** 회로와 하드웨어를 각각 독립적인 GATv2 인코더로 처리. 동일한 구조이지만 파라미터는 공유하지 않음 (서로 다른 도메인이므로).

**내용:**
- **GATv2 (Graph Attention Network v2)** 사용 — 이웃 노드에 대한 attention weight를 동적으로 학습
- 3-layer 구조, 각 layer:
  - GATv2 Attention (4 heads) → BatchNorm → ELU activation → Residual connection
- Circuit GNN과 Hardware GNN: 동일한 아키텍처, **별도 파라미터** (weight sharing 없음)
  - 이유: 회로 feature(게이트 수, depth 등)와 하드웨어 feature(T1, T2, error 등)는 완전히 다른 도메인

```
Input Features → Linear(input_dim, 64)
  → [GATv2 + BatchNorm + ELU + Residual] × 3 layers
  → Linear(64, 64) → Final Embedding
```

- Embedding dimension: 64
- Attention heads: 4 (각 head 16차원, concat 후 64차원)
- Edge feature가 attention score 계산에 통합됨
- 3-layer = 3-hop 이웃 정보 aggregation (3단계 떨어진 큐빗까지 정보 전파)

**다이어그램 제안:**
- 좌우 대칭 구조로 Circuit GNN / Hardware GNN 배치
- 각 GNN 내부를 3개 layer 블록으로 표현
- 출력: C (l×d), H (h×d) 행렬

**Speaker Notes:**
회로와 하드웨어 그래프를 각각 GATv2 기반 3-layer GNN으로 인코딩합니다. GATv2는 이웃 노드의 중요도를 동적으로 학습하는 graph attention 메커니즘입니다. 두 GNN은 구조는 같지만 파라미터는 별도입니다. 회로의 feature(게이트 수, depth)와 하드웨어의 feature(T1, error rate)는 완전히 다른 도메인이므로 weight를 공유하면 안 됩니다. 3-layer 구조를 통해 3-hop까지의 이웃 정보를 aggregation합니다.

---

### Slide 10: Model Architecture — Cross-Attention

**이 슬라이드의 목적:** 왜 단순 dot-product가 아닌 cross-attention이 필요한지, 그리고 어떻게 동작하는지 설명.

**핵심 메시지:** 좋은 매핑은 "유사한" 큐빗이 아니라 "상보적인" 큐빗의 매칭. Cross-attention이 이 상보적 관계를 포착.

**내용:**
- **왜 cross-attention인가?**
  - 단순 dot-product similarity는 "비슷한" 큐빗끼리 매칭
  - 하지만 좋은 매핑은 **상보적(complementary)** 관계:
    - 바쁜(gate 많은) 논리 큐빗 → 고품질(low error) 물리 큐빗에 배치해야 함
    - 이 둘은 feature 공간에서 "유사"하지 않지만, 좋은 매핑임
  - Cross-attention은 "이 논리 큐빗에게 어떤 물리 큐빗이 적합한가?"를 학습

- **구조:** 2 layers, 각 layer에서 양방향(bidirectional) attention
  - 회로 → 하드웨어 방향: C = LayerNorm(C + CrossAttn(Q=C, K=H, V=H))
    - "각 논리 큐빗이 어떤 물리 큐빗에 주목해야 하는가"
  - 하드웨어 → 회로 방향: H = LayerNorm(H + CrossAttn(Q=H, K=C, V=C))
    - "각 물리 큐빗이 어떤 논리 큐빗을 수용하기 적합한가"
  - 각 방향 후 FFN(Feed-Forward Network) + residual connection

- 결과: C'와 H'는 서로를 참조하여 "매핑에 최적화된" 임베딩으로 변환됨

**다이어그램 제안:**
- 두 임베딩(C, H) 사이의 양방향 화살표
- 각 방향에 "Query←→Key,Value" 관계 표시
- 2 layer를 세로로 쌓아서 반복 구조 표현

**Speaker Notes:**
Cross-attention은 이 모델의 핵심 설계입니다. 단순히 회로 큐빗과 하드웨어 큐빗의 임베딩을 비교하면, "비슷한" 것끼리 매칭하게 됩니다. 하지만 실제로 좋은 매핑은 상보적 관계입니다. 게이트가 많이 걸리는 바쁜 논리 큐빗은 오류율이 낮은 고품질 물리 큐빗에 배치해야 합니다. 이 둘은 feature 공간에서 비슷하지 않죠. Cross-attention을 통해 회로 임베딩이 하드웨어를 참조하고, 하드웨어 임베딩이 회로를 참조하면서, 매핑에 적합한 표현으로 변환됩니다.

---

### Slide 11: Model Architecture — Score Head, Sinkhorn, Hungarian

**이 슬라이드의 목적:** Cross-attention 이후 최종 매핑을 결정하는 과정 설명.

**핵심 메시지:** Score matrix → soft assignment(Sinkhorn) → hard assignment(Hungarian)의 3단계로, 미분 가능한 학습과 이산적 추론을 양립.

**내용:**

1. **Score Head:** 각 (논리 큐빗 i, 물리 큐빗 j) 쌍의 매핑 적합도 점수 계산
   - S_ij = (C'_i · W_q)^T · (H'_j · W_k) / √d_k
   - 결과: l×h 크기의 score matrix S

2. **Dummy Padding:** 논리 큐빗 수(l) < 물리 큐빗 수(h)이므로
   - (h-l)개의 dummy row(0으로 채움)를 추가하여 h×h 정방 행렬로 확장
   - Sinkhorn은 정방 행렬에서만 동작하므로 필요한 전처리

3. **Log-domain Sinkhorn:** Score matrix → doubly stochastic matrix (soft assignment)
   - 각 행/열의 합이 1이 되도록 반복 정규화
   - 온도 파라미터 τ: 높으면 uniform(부드러운), 낮으면 one-hot(날카로운)
   - Log-domain 구현: τ가 매우 낮을 때(≈0.05) 수치적 안정성 보장
   - **학습 시 사용:** 미분 가능하므로 gradient 전달 가능

4. **Hungarian Algorithm (추론 시만):** soft assignment → discrete 1:1 매핑
   - Sinkhorn 출력에서 최적의 이산적 할당을 결정
   - 상위 l개 행만 사용 (dummy row의 할당은 폐기)
   - **학습 시 불사용:** 미분 불가능하므로

**다이어그램 제안:**
- 3단계 변환을 시각적으로: Score Matrix(히트맵) → Sinkhorn(부드러운 히트맵) → Hungarian(이진 행렬)
- τ의 역할을 보여주는 작은 보조 그림 (high τ vs low τ)

**Speaker Notes:**
Cross-attention 이후 Score Head에서 모든 논리-물리 큐빗 쌍의 적합도 점수를 계산합니다. 이 점수를 Sinkhorn 알고리즘으로 doubly stochastic matrix, 즉 soft assignment로 변환합니다. 학습 시에는 이 soft assignment에서 바로 loss를 계산하고, 추론 시에는 Hungarian 알고리즘으로 이산적 1:1 매핑으로 변환합니다. 이 설계가 중요한 이유는, Sinkhorn까지는 미분 가능하므로 학습이 가능하고, Hungarian은 추론 시에만 사용하여 실제 할당을 결정하기 때문입니다.

---

### Slide 12: Training Strategy — Overview & Motivation

**이 슬라이드의 목적:** 2-Stage 학습 전략의 전체 그림과 "왜 이렇게 해야 하는가"의 동기 설명. 가장 논리적 설득력이 필요한 슬라이드.

**핵심 메시지:** Non-differentiable barrier 때문에 PST를 직접 최적화할 수 없으므로, (1) 먼저 기존 label로 기본 매핑 감각을 학습하고, (2) 그 위에서 미분 가능한 surrogate loss로 fine-tuning하는 2단계 전략을 사용.

**내용:**

**전체 그림:**
```
Stage 1: Supervised Pre-training (labeled data만 사용)
  Phase 1: MLQD + QUEKO 혼합 → CE Loss → 기본 매핑 감각 학습
  Phase 2: QUEKO only → CE Loss → true optimal로 정밀 보정
                    ↓
Stage 2: Unsupervised Surrogate Fine-tuning (전체 데이터 사용)
  L_adj + β·L_hop + α·L_node → noise-aware 최적화
```

**왜 Stage 1 (Supervised)이 반드시 필요한가?**
- Stage 2의 surrogate loss만으로는 충분하지 않은 3가지 이유:
  1. **Cold start 문제:** 랜덤 초기화 상태에서 surrogate gradient만으로는 "좋은 매핑"의 방향을 처음부터 찾기 어려움. 탐색 공간이 너무 넓음 (n! 가능한 매핑)
  2. **Surrogate ≠ PST:** Surrogate loss는 PST와 상관되도록 설계했지만 PST 자체가 아님. 좋은 초기점 없이 surrogate만 최적화하면, surrogate는 감소하는데 실제 PST는 개선되지 않는 overfitting 위험
  3. **Trivial solution 위험:** Surrogate loss는 "자주 상호작용하는 큐빗을 인접하게 배치하라"만 지시. 의미 있는 사전 지식 없이는 의미 없는 local minimum에 수렴할 수 있음

- **Stage 1의 역할:** "좋은 매핑이 대략 어떤 구조인지" 기본기를 label로 학습
- **Stage 2의 역할:** 그 기본기 위에서 실전 환경(SABRE routing + real noise)에 맞게 fine-tuning
- 비유: Stage 1 = 교과서로 기본기 학습, Stage 2 = 실전 문제로 응용력 향상

**다이어그램 제안:**
- 2단계를 세로 화살표로 연결
- 각 단계의 입력 데이터, loss, 목적을 박스로 표시
- "왜 필요한가" 부분은 Stage 1 옆에 보조 텍스트 박스로

**Speaker Notes:**
학습 전략의 핵심 동기는 non-differentiable barrier입니다. 모델 출력인 layout이 Qiskit transpile을 거쳐야 PST를 알 수 있는데, transpile은 미분 불가능합니다. 따라서 PST를 직접 최적화할 수 없습니다. 이 문제를 2단계로 해결합니다. Stage 1에서는 기존에 있는 정답 label(OLSQ2 solver 결과, QUEKO optimal)로 "좋은 매핑의 기본 구조"를 학습합니다. Stage 2에서는 label 없이도 계산 가능한 미분 가능한 surrogate loss로, noise를 고려한 fine-tuning을 합니다. Stage 1 없이 Stage 2만 하면, 랜덤 초기화 상태에서 surrogate loss의 gradient만으로는 방향을 잡기 어렵고, surrogate와 실제 PST의 괴리로 overfitting될 위험이 있습니다.

---

### Slide 13: Stage 1 — Supervised Pre-training (상세)

**이 슬라이드의 목적:** Stage 1의 구체적 동작과 2개 Phase의 이유 설명.

**핵심 메시지:** Phase 1에서 대량 데이터로 기본기를 잡고, Phase 2에서 소량이지만 절대 정답인 데이터로 정밀 보정.

**내용:**

**Loss:** Cross-Entropy
- 모델 출력 P (Sinkhorn의 doubly stochastic matrix)와 정답 permutation matrix 간의 차이를 최소화

**Phase 1: MLQD + QUEKO 혼합 (3,846 labeled circuits)**
- MLQD labels (3,729개): OLSQ2 solver의 결과 회로에서 역추적하여 추출
  - OLSQ2는 SMT solver로 layout과 routing을 동시에 최적화
  - 결과 회로에서 SWAP 패턴(CNOT 3개의 특정 패턴)을 역순으로 추적하여 initial layout 복원
  - 주의: 이 label은 OLSQ2의 routing과 함께 최적인 것이지, SABRE routing과는 불일치 가능
- QUEKO labels (540개): τ⁻¹ true optimal (SWAP 0개로 매핑 가능하도록 역설계된 벤치마크)
- Sinkhorn τ annealing: 1.0 → 0.05 (학습 초반 부드러운 assignment → 후반 날카로운 assignment)

**Phase 2: QUEKO only (486 circuits)**
- QUEKO만으로 fine-tuning, LR을 Phase 1의 1/10로 감소
- 왜 2단계로 나누는가?
  - MLQD labels: 양이 많지만(3,729개) **OLSQ2 router에 최적화된 label** → SABRE와 불일치 가능 (router-specific bias)
  - QUEKO labels: 양이 적지만(540개) **어떤 router를 쓰든 최적인 절대 정답** (zero-SWAP 보장)
  - Phase 1에서 MLQD의 대량 데이터로 기본 감각을 배우고, Phase 2에서 QUEKO의 router-agnostic 정답으로 보정
- LR을 낮추는 이유: Phase 1에서 학습한 내용을 크게 흐트리지 않으면서 정밀 조정

**다이어그램 제안:**
- Phase 1 → Phase 2 화살표
- 각 Phase의 데이터 소스, 양, 특성을 표로 비교
- τ annealing을 작은 그래프로 (x축: epoch, y축: τ)

**Speaker Notes:**
Stage 1은 두 Phase로 나뉩니다. Phase 1에서는 MLQD와 QUEKO의 labeled data를 모두 사용합니다. MLQD의 label은 OLSQ2라는 SMT solver의 결과 회로에서 역추적하여 추출한 것인데, 이 label은 OLSQ2의 routing 결정과 함께 최적인 것이라 SABRE routing과는 불일치가 있을 수 있습니다. 반면 QUEKO의 label은 SWAP이 0개인 토폴로지 기반 최적으로, 어떤 router를 쓰든 정답입니다. Phase 2에서 QUEKO만으로 fine-tuning하여 router-specific bias를 보정합니다.

---

### Slide 14: Stage 2 — Unsupervised Surrogate Fine-tuning (상세)

**이 슬라이드의 목적:** Stage 2의 surrogate loss 4가지를 상세 설명.

**핵심 메시지:** Label 없이도 "좋은 매핑"의 신호를 제공하는 미분 가능한 surrogate loss 설계. SABRE routing의 실제 의사결정 기준(인접 여부)에 직접 맞춘 loss + noise-aware 보조 loss.

**내용:**

**데이터:** 6,887 전체 회로 (label 불필요 → labeled + unlabeled 모두 사용)

**Total Loss = L_adj + β · L_hop + α · L_node**

| Loss | 가중치 | 역할 |
|------|:------:|------|
| L_adj | 1.0 (고정) | 핵심: 인접 매핑 유도 |
| L_hop | β = 0.2 | 보조: 비인접 배치 간 거리 구분 |
| L_node | α = 0.3 | 보조: 고품질 큐빗 배치 |

**L_adj (Adjacency Matching Loss) — 핵심 loss:**
- 목적: 논리적으로 연결된 큐빗 쌍이 물리적으로 **인접한(directly connected)** 큐빗에 매핑되도록 유도
- 핵심 아이디어: 인접한 물리 큐빗 쌍은 SWAP 없이 2-qubit gate 실행 가능 → SWAP 필요성을 직접 측정
- 작동 방식:
  - 하드웨어 coupling map에서 **binary adjacency matrix** A_hw 구성 (인접=1, 비인접=0)
  - 각 논리 엣지 (i,j)에 대해: P를 통해 기대되는 인접 확률 계산
  - **gate-frequency weighting**: 2-qubit gate가 많은 엣지에 더 높은 가중치 부여 (SWAP 비용이 더 크므로)
- L_adj ∈ [-1, 0]: -1이면 모든 상호작용 큐빗 쌍이 인접 (최적, SWAP 0개)
- **왜 error-weighted distance가 아닌 binary adjacency인가?**
  - SABRE routing의 실제 의사결정 = "인접한가 아닌가" (binary)
  - SABRE는 error-weighted 최단 경로를 따르지 않음 → error-weighted distance는 SABRE와 무관
  - Binary 신호가 SABRE의 실제 동작에 정확히 대응

**L_hop (Hop Distance Tiebreaker Loss) — L_adj 보조:**
- 목적: L_adj가 구분하지 못하는 비인접 배치 간의 차이를 포착
- L_adj는 binary(인접 vs 비인접)이므로, distance-2와 distance-10 배치를 구분하지 못함
- L_hop은 normalized hop distance를 사용하여 비인접 배치 중에서도 가까운 쪽을 선호하도록 유도
- L_hop ∈ [0, 1]: 0이면 모든 쌍이 최소 거리
- L_adj가 주도하고, L_hop은 **tiebreaker** 역할 (β = 0.2로 낮은 가중치)

**L_node (NISQ Node Quality Loss):**
- 목적: "바쁜" 논리 큐빗을 "고품질" 물리 큐빗에 매핑
- 작동 방식:
  - 각 물리 큐빗의 품질 점수를 **learnable 2-layer MLP**로 계산:
    q_score = sigmoid(MLP(hw_features))
    MLP: Linear(7→16) → ELU → Linear(16→1)
  - 입력 7개: T1, T2, readout_error, single_qubit_error, degree, t1_cx_ratio, t2_cx_ratio
  - MLP이므로 noise feature 간 **비선형 관계** 포착 가능 (예: T1×readout_error 상호작용)
  - gate가 많은 논리 큐빗이 q_score가 높은 물리 큐빗에 배치되도록 유도
- L_node ∈ [-1, 0]: qubit importance를 probability distribution으로 정규화 (합=1) → 회로 크기 무관하게 스케일 일정

**모든 loss term이 per-pair/per-qubit normalized:** 회로/하드웨어 크기에 상관없이 스케일이 비교 가능하도록 설계.

**Multi-programming:** 여러 회로를 하나의 disconnected circuit graph로 합쳐서 처리. L_adj의 intra-circuit clustering이 자연스럽게 회로 간 분리를 만들므로 별도의 separation loss 불필요.

**다이어그램 제안:**
- 3개 loss를 시각적으로 분리하여 각각의 역할 도식화
  - L_adj: 회로 그래프의 엣지 → 하드웨어의 인접 관계 대응 (binary: 연결됨/안됨)
  - L_hop: 비인접 배치의 hop distance 비교 (distance-2 vs distance-5, 가까울수록 좋음)
  - L_node: 논리 큐빗(바쁜)와 물리 큐빗(고품질) 간의 매칭 그림
- L_adj를 가장 크게, L_hop을 L_adj에 붙여서 보조 관계 표현

**Speaker Notes:**
Stage 2에서는 label 없이도 계산 가능한 3가지 surrogate loss를 사용합니다. 핵심인 L_adj는 SABRE routing의 실제 의사결정 기준에 직접 맞춘 loss입니다. SABRE는 인접한 큐빗 쌍이면 바로 gate를 실행하고, 아니면 SWAP을 삽입합니다. 이 binary 판단을 그대로 loss에 반영합니다. Gate가 많은 엣지에는 더 높은 가중치를 주어, SWAP 비용이 큰 부분을 우선 최적화합니다. L_hop은 L_adj의 보조로, 비인접 배치 중에서도 가까운 쪽을 선호하게 합니다. L_node는 바쁜 논리 큐빗을 고품질 물리 큐빗에 배치하는데, 기존 linear combination이 아닌 2-layer MLP로 noise feature 간 비선형 관계를 포착합니다. 예를 들어 T1이 길더라도 readout error가 크면 품질이 낮다는 식의 상호작용을 학습할 수 있습니다. Multi-programming은 여러 회로를 하나의 disconnected circuit으로 합쳐서 처리하며, L_adj의 intra-circuit clustering이 자연스럽게 회로 간 분리를 만듭니다.

---

### Slide 15: Dataset Overview

**이 슬라이드의 목적:** 사용하는 데이터셋의 전체 구성 요약.

**핵심 메시지:** 5개 데이터셋, 총 6,887 회로. Labeled 4,269개(Stage 1), Unlabeled 포함 전체(Stage 2).

**내용 (표):**

| Dataset | 회로 수 | Label 유무 | 사용 Stage | Label 출처 |
|---------|:-------:|:----------:|:----------:|-----------|
| QUEKO | 900 (540 labeled) | O (540) | Stage 1 + 2 | τ⁻¹ optimal (zero-SWAP 보장) |
| MLQD | 4,443 (3,729 labeled) | O (3,729) | Stage 1 + 2 | OLSQ2 solver 결과에서 역추적 |
| MQT Bench | 1,219 | X | Stage 2 only | — |
| QASMBench | 94 | X | Stage 2 only | — |
| RevLib | 231 | X | Stage 2 only | — |

- 총: 6,887 training circuits, 4,269 labels
- 전처리:
  - Gate normalization: 모든 회로를 {cx, id, rz, sx, x} 기본 게이트로 통일
  - Extreme circuit filtering: 엣지 1,000개 초과 회로 제거 (GNN 확장성)
  - Benchmark deduplication: 평가용 벤치마크 회로는 학습 데이터에서 제거

**Speaker Notes:**
5개 공개 데이터셋에서 총 6,887개 양자 회로를 사용합니다. 이 중 4,269개가 label을 가지고 있어 Stage 1 supervised learning에 사용되고, Stage 2에서는 label 유무와 관계없이 전체 데이터를 사용합니다. QUEKO는 회로 자체가 zero-SWAP으로 설계되어 절대적 정답 label을 제공하고, MLQD는 OLSQ2 solver의 결과에서 label을 추출했습니다. 나머지 3개 데이터셋은 label이 없어 Stage 2의 surrogate loss로만 학습됩니다.

---

### Slide 16: Hardware Backends

**이 슬라이드의 목적:** 학습/평가에 사용하는 하드웨어 구성 설명. 특히 "unseen backend에서 평가"라는 실험 설계의 의미 강조.

**핵심 메시지:** 60개 다양한 백엔드로 학습, 학습 시 본 적 없는 3개 백엔드로 평가 → 진정한 hardware-agnostic 성능 검증.

**내용:**

**Training: 60 backends**
- 55개 Qiskit FakeBackendV2 (5큐빗 ~ 127큐빗)
- 5개 synthetic backends (QUEKO/MLQD 데이터셋의 하드웨어 topology에 합성 noise 부여)
- 다양한 규모(5Q ~ 127Q)와 topology(선형, 격자, heavy-hex 등) 포함

**Test (UNSEEN): 3 backends**
- FakeToronto (27 큐빗)
- FakeBrooklyn (65 큐빗)
- FakeTorino (133 큐빗)
- 학습 시 한 번도 본 적 없는 하드웨어
- 다양한 규모(27Q, 65Q, 133Q)로 일반화 능력 평가

- 의미: 모델이 특정 하드웨어를 "외운" 것이 아니라, 하드웨어의 구조적 패턴(connectivity, noise 분포)을 학습했는지 검증
- Native 2-qubit gate 종류(cx / ecr / cz)도 백엔드마다 다르지만, 자동 감지하여 처리

**다이어그램 제안:**
- Training backends를 큐빗 수 기준으로 분포 표시 (히스토그램 또는 점 그래프)
- Test backends 3개를 별도 색상으로 강조, "UNSEEN" 라벨 표시
- 규모가 다양함을 시각적으로 보여주기

**Speaker Notes:**
60개의 다양한 하드웨어 백엔드로 학습하고, 학습 때 한 번도 본 적 없는 3개 백엔드에서 평가합니다. 이 실험 설계가 중요한 이유는, 모델이 특정 하드웨어를 외운 것이 아니라 하드웨어의 구조적 패턴을 일반적으로 학습했는지를 검증하기 때문입니다. 테스트 백엔드는 27큐빗, 65큐빗, 133큐빗으로 다양한 규모에서 일반화 능력을 평가합니다.

---

### Slide 17: Evaluation Protocol

**이 슬라이드의 목적:** 어떻게 성능을 측정하고 비교하는지 설명.

**핵심 메시지:** PST를 noise simulation으로 측정하고, 6가지 기존 방법과 비교.

**내용:**

**Primary Metric: PST (Probability of Successful Trials)**
- 양자 회로를 noise simulation으로 실행하여, 정답 출력이 나올 확률을 측정
- Qiskit Aer, tensor_network simulator + GPU (cuQuantum)
- 8,192 shots (충분한 통계적 신뢰도)
- optimization_level 3 (Qiskit 최대 최적화)

**Baselines (6가지 기존 방법):**
- **SABRE:** Qiskit 기본. Heuristic 기반 SWAP 최소화
- **Trivial:** 순서대로 배치 (q0→p0, q1→p1, ...). 가장 단순한 baseline
- **Dense:** 하드웨어에서 연결이 밀집된 영역에 배치
- **NASSC:** Noise-Aware + SWAP Cost 통합 heuristic
- **Noise-Adaptive:** Noise 기반 subgraph 매칭
- **QAP:** Quadratic Assignment Problem 최적화

**Benchmark Circuits: 8개 standard circuits (3Q ~ 9Q)**
- toffoli_3, fredkin_3, 3_17_13, 4mod5-v1_22, mod5mils_65, alu-v0_27, decod24-v2_43, 4gt13_92
- 양자 컴파일 연구에서 널리 사용되는 표준 벤치마크

**Speaker Notes:**
성능은 PST로 측정합니다. 각 벤치마크 회로를 noise simulation으로 실행하여 정답이 나올 확률을 측정합니다. 6가지 기존 방법과 비교하는데, SABRE는 Qiskit의 기본 방법이고, Trivial은 아무 최적화 없이 순서대로 배치하는 가장 단순한 baseline입니다. NASSC과 Noise-Adaptive는 noise를 고려하는 방법이고, QAP은 최적화 기반 접근입니다. 8개의 표준 벤치마크 회로에서 평가합니다.

---

### Slide 18: Current Progress

**이 슬라이드의 목적:** 현재까지의 구현 상태와 진행 상황 보고.

**내용:**

**구현 완료:**
- 데이터 파이프라인: 5개 데이터셋 수집, 전처리, train/val split 관리
- 전체 모델 아키텍처: Dual GNN Encoder, Cross-Attention, Score Head, Sinkhorn, Hungarian
- 2-Stage 학습 루프: CE loss, surrogate losses, τ annealing, early stopping
- 평가 프레임워크: PST 측정, 6가지 baseline 비교, benchmark runner
- 테스트: 119개 unit/integration tests

**현재 단계:**
- 학습 실험 진행 중
- (실험 결과가 있으면 여기에 추가: 그래프, 수치 등)

**Speaker Notes:**
현재 전체 시스템 구현이 완료된 상태입니다. 데이터 파이프라인, 모델, 학습 루프, 평가 프레임워크가 모두 동작하며, 119개의 테스트로 검증되었습니다. 현재 학습 실험을 진행 중이며, 초기 결과를 분석하고 있습니다.

---

### Slide 19: Expected Contributions

**이 슬라이드의 목적:** 이 연구의 학술적 기여를 명확히 정리.

**내용:**

1. **Hardware-Agnostic Mapping:** 단일 모델이 학습 시 본 적 없는 다양한 NISQ 하드웨어에서 fine-tuning 없이 동작하는 최초의 학습 기반 접근
2. **Noise-Aware Surrogate Loss 설계:** Non-differentiable barrier를 우회하면서 noise를 고려하는 미분 가능한 대리 손실 함수 체계 (L_surr, L_node)
3. **Cross-Attention 기반 Circuit-Hardware 매칭:** 유사성이 아닌 상보적 관계를 포착하는 양방향 cross-attention 아키텍처
4. **Learnable Quality Score의 해석 가능성:** 학습된 noise factor 가중치(w1~w5) 분석을 통한 "어떤 noise factor가 PST에 중요한가"에 대한 해석 가능한 insight

**Speaker Notes:**
이 연구의 주요 기여는 네 가지입니다. 첫째, hardware-agnostic 학습 기반 매핑이라는 새로운 접근입니다. 둘째, non-differentiable barrier를 우회하면서 noise를 고려하는 surrogate loss 설계입니다. 셋째, circuit-hardware 간 상보적 관계를 포착하는 cross-attention 아키텍처입니다. 넷째, 학습된 noise factor 가중치를 통해 PST에 중요한 factor를 분석할 수 있는 해석 가능성입니다.

---

### Slide 20: Timeline / Next Steps

**이 슬라이드의 목적:** 향후 계획 공유.

**내용:**
- Stage 1 + Stage 2 학습 완료 및 하이퍼파라미터 튜닝
- Unseen backend (Toronto 27Q, Brooklyn 65Q, Torino 133Q)에서 PST 평가
- Ablation study: Cross-Attention 유무, L_node 기여도, τ annealing 전략 등
- 논문 작성
- (향후 확장) Multi-programming: 여러 양자 회로를 하나의 하드웨어에 동시 매핑

**Speaker Notes:**
향후 계획입니다. 현재 학습 실험을 진행 중이며, unseen backend에서의 PST 평가와 ablation study를 통해 각 설계 결정의 기여도를 검증할 예정입니다. 이후 논문 작성으로 이어가겠습니다.

---

### Slide 21: Q&A

**내용:** "감사합니다. 질문 있으시면 말씀해주세요."

---

## 디자인 가이드라인

- **색상 팔레트:** 파랑(주색) + 주황/노랑(강조). 학술 발표 톤.
- **폰트:** 본문은 sans-serif (예: Pretendard, Noto Sans KR), 코드/수식은 monospace
- **다이어그램 스타일:** 깔끔한 블록 다이어그램, 불필요한 장식 배제
- **슬라이드당 텍스트 최소화:** 가능하면 도식 위주, 상세 내용은 speaker notes에
- **핵심 포인트 강조:** 각 슬라이드에서 가장 중요한 1~2문장은 굵은 글씨 또는 색상으로 강조
- **일관된 레이아웃:** 제목 위치, 본문 영역, 하단 페이지 번호 통일
- **표:** 테두리는 최소화, 헤더 행만 색상 배경
- **화살표/흐름:** 왼→오 또는 위→아래 방향 통일

## 슬라이드 수 요약

총 21장. 발표 시간 약 30~40분 예상.
- Background & Motivation: 4장 (Slide 2~5)
- Pipeline & Architecture: 6장 (Slide 6~11)
- Training Strategy: 3장 (Slide 12~14)
- Data & Evaluation: 3장 (Slide 15~17)
- Progress & Contributions: 3장 (Slide 18~20)
- Title & Q&A: 2장 (Slide 1, 21)
