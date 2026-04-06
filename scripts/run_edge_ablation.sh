#!/bin/bash
# Edge ablation: C1 (edge 3dim) vs C2 (edge 5dim), 3 seeds each
# Runs 2 at a time: GPU 0 + GPU 1

set -e
cd /home/jaehee/workspace/projects/GraphQMap

echo "=== Pair 1: seed=42 ==="
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/stage2_sinkhorn_adj.yaml \
  --name circuit_edge3_seed42 \
  --override model.circuit_gnn.edge_input_dim=3 \
  --override seed=42 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/stage2_sinkhorn_adj.yaml \
  --name circuit_edge5_seed42 \
  --override seed=42 &
PID1=$!

wait $PID0 $PID1
echo "=== Pair 1 done ==="

echo "=== Pair 2: seed=43 ==="
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/stage2_sinkhorn_adj.yaml \
  --name circuit_edge3_seed43 \
  --override model.circuit_gnn.edge_input_dim=3 \
  --override seed=43 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/stage2_sinkhorn_adj.yaml \
  --name circuit_edge5_seed43 \
  --override seed=43 &
PID1=$!

wait $PID0 $PID1
echo "=== Pair 2 done ==="

echo "=== Pair 3: seed=44 ==="
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/stage2_sinkhorn_adj.yaml \
  --name circuit_edge3_seed44 \
  --override model.circuit_gnn.edge_input_dim=3 \
  --override seed=44 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/stage2_sinkhorn_adj.yaml \
  --name circuit_edge5_seed44 \
  --override seed=44 &
PID1=$!

wait $PID0 $PID1
echo "=== Pair 3 done ==="

echo "=== ALL 6 RUNS COMPLETE ==="
