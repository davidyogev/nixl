#!/usr/bin/env bash
# Single-host 2x2 run of test_ht.py against an EP install at /workspace/nixl/install.
# Intended for an RDMA-capable host (RoCE/IB present); see UCX_TLS note below.
#
# Topology (all on one physical host):
#   num_ranks            = 4
#   num_local_ranks      = 2    # 2 NVL peers per island
#   num_nodes            = 2    # via WORLD_SIZE; 2 logical RDMA islands
#   NUM_MAX_NVL_PEERS    = 2    # compile-time, kernels/configs.cuh
#
# Per-pair routing inside the kernel:
#   rank 0 <-> rank 1   : cuda_ipc  (NVL codepath; bare cudaIpcOpenMemHandle pointers)
#   rank 2 <-> rank 3   : cuda_ipc  (NVL codepath; bare cudaIpcOpenMemHandle pointers)
#   ranks across pairs  : nixlPut<WARP>  (RDMA codepath through NIXL's UCX backend)
#
# UCX_TLS=^cuda_ipc removes cuda_ipc from UCX's available transports so the RDMA
# endpoints land on a NIC-backed transport (rc_mlx5 et al). The NVL path uses
# CUDA's own IPC API (cudaIpcGetMemHandle / cudaIpcOpenMemHandle) and never
# touches UCX, so it is unaffected.
#
# Example end-to-end on the HPC compute node, inside a Pyxis container from
# lishapira/latest.sqsh with the patched source bind-mounted at /workspace/nixl:
#   srun -A <acct> --partition=<part> --nodes=1 --gres=gpu:4 \
#        --container-image=/lustre/.../dyogev/latest.sqsh \
#        --container-mounts=/lustre/.../dyogev/nixl:/workspace/nixl \
#        --pty bash
#   # inside the container, after meson+ninja install:
#   bash examples/device/ep/tests/run_test_ht_2x2.sh

set -euo pipefail

if [[ ! -f /workspace/nixl/install/lib/python3/dist-packages/nixl_ep/nixl_ep_cpp.cpython-312-x86_64-linux-gnu.so ]]; then
    echo "error: /workspace/nixl/install does not contain a built nixl_ep module." >&2
    echo "       Build first:  meson setup nixl_build --prefix=/workspace/nixl/install \\" >&2
    echo "                       -Ducx_path=/opt/hpcx/ucx -Dbuild_docs=false -Drust=false \\" >&2
    echo "                       -Dbuild_nixl_ep=true -Dlibfabric_path=/opt/amazon/efa \\" >&2
    echo "                       --buildtype=debug" >&2
    echo "                     ninja -C nixl_build install" >&2
    exit 1
fi

# Force UCX to use the NIC for cross-island endpoints instead of cuda_ipc.
# Override on the command line (e.g. UCX_TLS=... bash run_test_ht_2x2.sh) if a
# different transport mix is needed.
export UCX_TLS=${UCX_TLS:-^cuda_ipc}
export PYTHONPATH=/workspace/nixl/install/lib/python3/dist-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/workspace/nixl/install/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export NIXL_PLUGIN_DIR=/workspace/nixl/install/lib/x86_64-linux-gnu/plugins

# Pretend this single host is 2 logical nodes so init_dist computes num_nodes=2
# and the dispatch layout splits into 2 RDMA islands.
export WORLD_SIZE=2
export RANK=0
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-8361}

cd /workspace/nixl
exec python3 examples/device/ep/tests/test_ht.py \
    --num-processes 4 \
    --num-local-ranks 2 \
    "$@"
