# Session Handoff — 2026-04-13

## What was done this session

### Goal
"Noise-Aware Iterative Attention for Scalable Qubit Mapping" 논문의 핵심 인사이트를 GraphQMap에 적용.
논문 원문은 `/reference_paper_noise_aware_iterative_attention.md`에 저장됨.

### 3가지 변경 사항 구현 완료

#### 1. C_eff 통합 비용 행렬 (`data/hardware_graph.py`)
- `precompute_c_eff()` / `precompute_c_eff_synthetic()` 추가
- Floyd-Warshall edge weight = `3×ε₂` (SWAP = 3 CX gates)
- 인접 쌍은 raw `ε₂` (직접 gate, SWAP 없음)
- 기존 `d_error`는 raw `ε₂`만 사용해서 SWAP 3× overhead 미반영이었음
- Toronto 27Q 기준: C_eff range [0.006, 0.765] vs d_error [0.006, 0.255], 비인접 쌍 비율 = 3.0×

#### 2. QAP Fidelity Loss (`training/losses.py`)
- `@register_loss("qap_fidelity")` — `QAPFidelityLoss` 클래스
- `tr(Ã_c P C_eff P^T)` 계산: PC = P@C_eff, APC = Ã_c@PC, trace = (APC * P).sum()
- `circuit_adj` (Ã_c) dense matrix는 collation에서 `circuit_edge_pairs`/`circuit_edge_weights`로 구축
- normalize=True: (l + |E|)로 나누어 cross-batch 비교 가능

#### 3. Iterative Score Refinement (`models/graphqmap.py`)
- `forward()`에 mirror descent feedback loop 추가
- `S^(t+1) = S_norm - λ · Ã_c · P^(t) · C_eff`
- **Score normalization (critical)**: `S_norm = (S - mean) / std` — GNN이 mean=38, std=24의 큰 score를 만들어서, normalization 없이는 feedback (std=0.07)이 318× 작아 무시됨. Normalize 후 ratio 2.92로 개선.
- `refine_lambda`: `nn.Parameter` (learnable scalar, init=1.0)
- `refine_beta`: temperature decay per iteration (default 0.9)
- `refine_iterations=0`이면 기존과 100% 동일 (backward compatible)

### 데이터 파이프라인 배관

| 파일 | 변경 |
|---|---|
| `data/dataset.py` | `c_eff` 필드 + `_c_eff_cache` + `_get_c_eff()` + `circuit_adj` 행렬 빌드 in collation |
| `training/trainer.py` | `c_eff`, `circuit_adj`를 model.forward()와 loss_kwargs 양쪽에 전달 |
| `evaluate.py` | 추론 시 `precompute_c_eff` import + `c_eff`/`circuit_adj` 계산하여 model.predict()에 전달 |
| `train.py` | PST validation에서 `precompute_c_eff` + `extract_circuit_features` import, `c_eff`/`circuit_adj` 전달 |

### 새 파일

| 파일 | 내용 |
|---|---|
| `configs/stage2_qap_refine.yaml` | Full backend config (test: Toronto/Rochester/Washington) |
| `configs/stage2_toronto_qap_refine.yaml` | Toronto-only config (test: Toronto/Rochester/Washington) |
| `reference_paper_noise_aware_iterative_attention.md` | 원본 논문 Section 1~3 전문 |
| `SESSION_HANDOFF_20260413.md` | 이 파일 |

### 문서 업데이트

- `CLAUDE.md`: Phase 6 섹션 추가 (QAP mirror descent, v1/v2 결과, score normalization)
- `docs/RESEARCH_SPEC.md`: Section 4.6 (Iterative Score Refinement), qap_fidelity loss, Appendix A.2b (C_eff)

### Trainer 수정 (user가 직접 수정)
- `training/trainer.py`: `best_loss` 트래킹 + `best_loss` checkpoint 저장 추가 (line 523~532)

### Config 수정 (user가 직접 수정)
- `configs/stage2_toronto_qap_refine.yaml`: test backends → Toronto/Rochester/Washington
- `configs/stage2_qap_refine.yaml`: training에 FakeBrooklyn/FakeTorino 추가, test → Toronto/Rochester/Washington

---

## 실험 결과

### v1: Score normalization 없음
- Run: `runs/stage2/20260413_040016_toronto_qap_refine_v1`
- Config: `stage2_toronto_qap_refine.yaml`, Toronto-only, qap_fidelity loss, iterations=5
- Best Val PST: 0.6664 (epoch 44)
- **Eval Toronto OURS+SABRE 0.3216, OURS+NASSC 0.3626 — SABRE(0.3550)보다 낮음**
- 원인: S^(0) mean=38 vs feedback max=0.42 → ratio 318× → refinement이 layout에 영향 없음
- QAP loss 단독으로는 GNN에 충분한 gradient signal 제공 못함

### v2: Score normalization 적용 (현재 코드 상태)
- Run: `runs/stage2/20260413_050340_toronto_qap_refine_v2_normscore`
- 동일 config, S^(0)를 z-score normalize한 것만 다름
- Best Val PST: **0.7954** (epoch 9)
- **Eval Toronto OURS+SABRE 0.5116, OURS+NASSC 0.5369 — QAP+NASSC(0.5351)과 대등**
- bv_n3(0.90), bv_n4(0.85), xor5(0.93)에서 QAP 능가
- toffoli_3(0.47 vs QAP 0.59), peres_3(0.70 vs QAP 0.78)은 아직 약함
- Val PST 진동 심함 (0.79→0.60→0.77→0.59), best가 epoch 9에 너무 일찍 등장

---

## 남은 작업 / 다음 방향

### 즉시 가능한 실험
1. **Multi-backend 학습**: `configs/stage2_qap_refine.yaml`로 full backend 학습. Toronto-only에서 대등했으니 multi-backend에서 개선 기대
2. **Hyperparameter 태닝**: iterations (3,5,10), lambda_init (0.5,1.0,2.0), beta (0.8,0.9,0.95)
3. **QAP loss + adjacency 조합**: qap_fidelity 단독 대신 기존 adjacency_size_aware와 결합 테스트

### 구조적 개선
4. **Val PST 진동 문제**: best PST가 epoch 9에 등장하고 이후 하락. tau annealing이 너무 공격적이거나, QAP loss의 gradient가 후반에 너무 커질 수 있음
5. **Routing 통합**: 논문의 핵심 중 하나인 predecessor matrix 기반 noise-optimal routing 구현 (현재는 SABRE/NASSC 외부 router 사용)

### 참고: 기존 baseline 대비
- 이전 Toronto-only best (adj=0.3 baseline): OURS+NASSC ~0.60 (CLAUDE.md Phase 6 v1 results 기록)
- v2 QAP refine: OURS+NASSC 0.5369 — baseline보다는 낮지만 QAP와 대등
- 이전 best는 val=test leakage 포함된 수치라 직접 비교 주의

---

## 테스트 상태
- 152개 테스트 모두 통과 (마지막 확인: 구현 직후)
- `trainer.py` user 수정 이후 테스트 재실행 안 함 — 다음 세션에서 확인 권장
