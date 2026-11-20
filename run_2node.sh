#!/bin/bash
# 2노드 8GPU: 2노드 앱의 호스트 셸(ssh $(hostname))에서 실행
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SIF=/home01/$USER/images/pytorch-vscode-25.03-gh200-r1-aarch64.sif
JOB=${JOB:-$(squeue --all -h -u "$USER" -w "$(hostname -s)" -o %i | head -n1)}
[ -n "$JOB" ] || { echo "이 노드에서 실행 중인 내 작업이 없습니다. VS Code 앱 노드에서 실행하세요."; exit 2; }

module purge
module load apptainer/1.4.5

NODES=$(squeue --all -h -j "$JOB" -o %N)
NNODES=$(scontrol show hostnames "$NODES" | wc -l)
MASTER=$(scontrol show hostnames "$NODES" | head -n1)
PORT=$((20000 + (JOB + RANDOM) % 20000))
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT OMP_NUM_THREADS=8
echo "job=$JOB nnodes=$NNODES nodes=$NODES master=$MASTER:$PORT"

srun --jobid="$JOB" --overlap --nodes="$NNODES" --ntasks-per-node=1 --cpus-per-task=288 \
     --mpi=none --kill-on-bad-exit=1 --gpu-bind=none \
  apptainer --quiet exec --nv --bind "$HERE" --pwd "$HERE" "$SIF" \
  .venv/bin/torchrun --nnodes="$NNODES" --nproc-per-node=4 \
     --rdzv-backend=c10d --rdzv-id="$JOB-$PORT" --rdzv-endpoint="$MASTER:$PORT" --max-restarts=0 \
     train_ddp.py "$@" 2>&1 |

  awk '/ NCCL INFO /{ if (/Selected provider is cxi/) n++; next } { print; fflush() }
       END { print (n ? "\n노드 간 통신: Slingshot (cxi)" : "\n노드 간 통신: Slingshot 아님. 설정을 확인하세요") }'
