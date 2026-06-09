#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-node launcher for the NVLink fault-tolerance (contraction) test in a
# 3-node x 2-GPU topology (6 total ranks). The single-node launcher
# (run_nvlink_fault_tolerance_test.sh) spawns every worker inside one
# torch.multiprocessing.spawn on the local box, which only exercises intra-node
# NVLink. This script instead fans out to 3 nodes via srun so the kill of
# ranks {0,1} actually models loss of every GPU on the first physical node.
#
# How it works:
#   - Must be invoked from inside an existing SLURM allocation of >=3 nodes
#     with >=2 GPUs each (no --container-image on the outer srun -- the
#     launcher itself runs on bare metal so it has access to SLURM_JOB_ID,
#     SLURM_JOB_NODELIST, and the cluster's `srun` binary).
#   - elastic.py is fundamentally rank-server-driven: workers ask the rank
#     server for the next global rank in connection order. The master node is
#     started first (no --tcp-server), gets a head-start so its 2 workers
#     register as ranks 0 and 1, then the other 2 nodes are launched with
#     --tcp-server <master_ip> so their 4 workers register as ranks 2..5.
#   - That ordering is what pins the kill of ranks {0,1} (declared in
#     nvlink_fault_tolerance_3node.json) to the master node, simulating
#     whole-node loss.
#   - Each per-node invocation runs inside the container via
#     `srun --overlap --container-image=... --container-mounts=...`. The env
#     setup (NIXL_INSTALL/PYTHONPATH/LD_LIBRARY_PATH/NIXL_PLUGIN_DIR + unset
#     UCX_TLS) is sourced from a small helper script we drop on the shared
#     lustre mount so every node sees the exact same setup.
#   - After the test completes (or fails), a cleanup report is collected from
#     every node and written next to the per-node logs.
#
# Typical usage (from a master-node bare-metal shell inside the allocation):
#
#   # Smoke: validates multi-node coordination on the 6-rank baseline plan
#   # (no kills). Run this FIRST whenever the launcher or environment changes.
#   bash run_nvlink_fault_tolerance_3node.sh --smoke
#
#   # Real fault run: kills ranks 0,1 (= node 0) before dispatch.
#   bash run_nvlink_fault_tolerance_3node.sh --fault-kill-timing before-dispatch
#
#   # Both kills targeted in different timings (one per victim in ascending
#   # rank-id order, so rank 0 -> first timing, rank 1 -> second timing).
#   bash run_nvlink_fault_tolerance_3node.sh \
#       --fault-kill-timing before-dispatch after-dispatch
#
# Site overrides via env (defaults match dyogev's lustre layout):
#   IMAGE                    container squashfs path
#   LUSTRE_NIXL              host-side nixl tree to mount
#   CTR_NIXL                 container-side mount target (must be visible to
#                            all 3 nodes via the same lustre mount)
#   WAIT_AFTER_MASTER_SECS   how long to wait after starting the master before
#                            launching the worker nodes; covers container
#                            cold-start + rank-server bind + 2 master-side
#                            worker registrations
#   SETTLE_SECONDS           wait between test end and cleanup queries
#
# Exit code:
#   - 0 only if every node's elastic.py exited 0 (or, for the fault plan, with
#     the expected fault signal).
#   - Otherwise the rc of the first failing node.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Required SLURM environment
# ---------------------------------------------------------------------------
: "${SLURM_JOB_ID:?must run inside a SLURM allocation (SLURM_JOB_ID unset). Example: 'srun -N 3 --gpus-per-node=2 --pty bash' on the login node, then run this script from the resulting shell.}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST unset}"

if ! command -v scontrol >/dev/null 2>&1; then
    echo "error: scontrol not found; this launcher must run from a host that has SLURM client tools available." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Site config (defaults match the dyogev personal lustre layout)
# ---------------------------------------------------------------------------
NODES_REQUIRED=3
GPUS_PER_NODE=2
TOTAL_RANKS=$((NODES_REQUIRED * GPUS_PER_NODE))

IMAGE=${IMAGE:-/lustre/fsw/portfolios/network/projects/network_research_advdev/users/dyogev/latest.sqsh}
LUSTRE_NIXL=${LUSTRE_NIXL:-/lustre/fsw/portfolios/network/projects/network_research_advdev/users/dyogev/nixl}
CTR_NIXL=${CTR_NIXL:-/workspace/nixl}
MOUNT_SPEC="${LUSTRE_NIXL}:${CTR_NIXL}"

ELASTIC_DIR_CTR="${CTR_NIXL}/examples/device/ep/tests/elastic"
WAIT_AFTER_MASTER_SECS=${WAIT_AFTER_MASTER_SECS:-25}
SETTLE_SECONDS=${SETTLE_SECONDS:-5}

# ---------------------------------------------------------------------------
# 2. Parse args
# ---------------------------------------------------------------------------
SMOKE=0
PLAN_FILE=""
RESULTS_DIR_HOST=""
declare -a EXTRA_ARGS=()

usage() {
    sed -n '2,/^set -euo pipefail$/p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)       SMOKE=1; shift;;
        --plan)        PLAN_FILE="$2"; shift 2;;
        --results-dir) RESULTS_DIR_HOST="$2"; shift 2;;
        -h|--help)     usage; exit 0;;
        --)            shift; while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done; break;;
        *)             EXTRA_ARGS+=("$1"); shift;;
    esac
done

if [[ -z "$PLAN_FILE" ]]; then
    if [[ "$SMOKE" -eq 1 ]]; then
        PLAN_FILE="nvlink_fault_tolerance_3node_baseline.json"
    else
        PLAN_FILE="nvlink_fault_tolerance_3node.json"
    fi
fi
PLAN_FILE_CTR="${ELASTIC_DIR_CTR}/${PLAN_FILE}"

# ---------------------------------------------------------------------------
# 3. Discover nodes
# ---------------------------------------------------------------------------
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if (( ${#NODES[@]} < NODES_REQUIRED )); then
    echo "error: need ${NODES_REQUIRED} nodes, allocation has ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi
MASTER_NODE="${NODES[0]}"
WORKER_NODES=("${NODES[@]:1:$((NODES_REQUIRED - 1))}")

# ---------------------------------------------------------------------------
# 4. Results dir (on lustre, visible from every node via $MOUNT_SPEC)
# ---------------------------------------------------------------------------
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
if [[ -z "$RESULTS_DIR_HOST" ]]; then
    RESULTS_DIR_HOST="${LUSTRE_NIXL}/examples/device/ep/tests/elastic/results_3node/${RUN_ID}"
fi
mkdir -p "$RESULTS_DIR_HOST"
RESULTS_DIR_CTR="${CTR_NIXL}/examples/device/ep/tests/elastic/results_3node/$(basename "$RESULTS_DIR_HOST")"

LAUNCHER_LOG="$RESULTS_DIR_HOST/launcher.log"
# Mirror everything to launcher.log AND keep it on stdout.
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] [launcher] $*"; }

say "===== launcher start ====="
say "job=${SLURM_JOB_ID}"
say "nodes(${#NODES[@]})=${NODES[*]}"
say "master_node=${MASTER_NODE}"
say "worker_nodes=${WORKER_NODES[*]}"
say "image=${IMAGE}"
say "mount=${MOUNT_SPEC}"
say "plan(host)=${LUSTRE_NIXL}/examples/device/ep/tests/elastic/${PLAN_FILE}"
say "plan(ctr) =${PLAN_FILE_CTR}"
say "smoke=${SMOKE} gpus_per_node=${GPUS_PER_NODE} total_ranks=${TOTAL_RANKS}"
say "wait_after_master_secs=${WAIT_AFTER_MASTER_SECS}"
say "extra_args=${EXTRA_ARGS[*]:-(none)}"
say "results_dir(host)=${RESULTS_DIR_HOST}"
say "results_dir(ctr) =${RESULTS_DIR_CTR}"

# ---------------------------------------------------------------------------
# 5. Per-node startup helper (env + exec). Lives on lustre so every node sees
#    the exact same script via $MOUNT_SPEC.
# ---------------------------------------------------------------------------
NODE_SCRIPT_HOST="$RESULTS_DIR_HOST/node_startup.sh"
NODE_SCRIPT_CTR="$RESULTS_DIR_CTR/node_startup.sh"
cat > "$NODE_SCRIPT_HOST" <<'NODESCRIPT'
#!/usr/bin/env bash
# Per-node startup: sets up NIXL EP env in-container, then execs python3 with
# the args passed to this script. Kept tiny and self-contained so the launcher
# can dispatch identical commands to every node via a single srun line.
set -euo pipefail
export NIXL_INSTALL=${NIXL_INSTALL:-/workspace/nixl/install}
NIXL_EP_CPP=$(ls "${NIXL_INSTALL}"/lib/python3/dist-packages/nixl_ep/nixl_ep_cpp.cpython-*.so 2>/dev/null | head -n 1 || true)
if [[ -z "${NIXL_EP_CPP}" ]]; then
    echo "[node=$(hostname -s)] error: ${NIXL_INSTALL} does not contain a built nixl_ep module." >&2
    echo "[node=$(hostname -s)] (Re)build first: meson setup nixl_build --prefix=${NIXL_INSTALL} -Ducx_path=/opt/hpcx/ucx -Dbuild_docs=false -Drust=false -Dbuild_nixl_ep=true -Dlibfabric_path=/opt/amazon/efa --buildtype=release && ninja -C nixl_build install" >&2
    exit 1
fi
export PYTHONPATH=${NIXL_INSTALL}/lib/python3/dist-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${NIXL_INSTALL}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export NIXL_PLUGIN_DIR=${NIXL_INSTALL}/lib/x86_64-linux-gnu/plugins
unset UCX_TLS
export PYTHONUNBUFFERED=1
echo "[node=$(hostname -s)] $(date -u +%Y-%m-%dT%H:%M:%SZ) startup: python3 $*"
echo "[node=$(hostname -s)] NIXL_INSTALL=${NIXL_INSTALL}"
echo "[node=$(hostname -s)] cuda_devices=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | wc -l)"
exec python3 "$@"
NODESCRIPT
chmod +x "$NODE_SCRIPT_HOST"
say "wrote per-node startup script: ${NODE_SCRIPT_HOST}"

# ---------------------------------------------------------------------------
# 6. Resolve master IP that workers will use to reach the rank server.
#    We ask the master node itself for `hostname -I` and take the first IP.
#    This typically returns the primary eth/IB IP, which is reachable from
#    siblings in the same allocation. If your cluster routes the rank server
#    traffic over a specific fabric, override by exporting MASTER_IP before
#    calling this script.
# ---------------------------------------------------------------------------
if [[ -z "${MASTER_IP:-}" ]]; then
    say "resolving master IP from ${MASTER_NODE} via 'hostname -I'..."
    MASTER_IP=$(srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$MASTER_NODE" \
        bash -c 'hostname -I | awk "{print \$1}"' 2>/dev/null | tr -d '[:space:]' || true)
fi
if [[ -z "$MASTER_IP" ]]; then
    echo "error: could not resolve master IP for node ${MASTER_NODE}" >&2
    exit 1
fi
say "master_ip=${MASTER_IP}"

# ---------------------------------------------------------------------------
# 7. Helper: dispatch an elastic.py invocation to a specific node, in
#    container, in background. Returns the bg pid via echo.
# ---------------------------------------------------------------------------
run_elastic_on() {
    local node="$1" log_name="$2"
    shift 2
    local log_file="${RESULTS_DIR_HOST}/${log_name}"
    say "dispatching to ${node} (log=${log_file})"
    say "  cmd: srun ... -w ${node} bash ${NODE_SCRIPT_CTR} ${ELASTIC_DIR_CTR}/elastic.py --plan ${PLAN_FILE_CTR} --num-processes ${GPUS_PER_NODE} $*"
    srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$node" \
        --container-image="$IMAGE" \
        --container-mounts="$MOUNT_SPEC" \
        --container-workdir="$CTR_NIXL" \
        bash "$NODE_SCRIPT_CTR" \
            "$ELASTIC_DIR_CTR/elastic.py" \
            --plan "$PLAN_FILE_CTR" \
            --num-processes "$GPUS_PER_NODE" \
            "$@" \
        > "$log_file" 2>&1 &
    echo $!
}

# ---------------------------------------------------------------------------
# 8. Start master (no --tcp-server -- it hosts TCPStore + rank server).
# ---------------------------------------------------------------------------
say "starting MASTER on ${MASTER_NODE}..."
MASTER_PID=$(run_elastic_on "$MASTER_NODE" "master_${MASTER_NODE}.log" "${EXTRA_ARGS[@]}")
say "master pid (background srun) = ${MASTER_PID}"

# Give the master a head start: container cold-start + rank-server bind +
# rank 0,1 registration. If this is too short the worker nodes will steal
# ranks 0,1 and the test will kill the wrong physical node.
say "waiting ${WAIT_AFTER_MASTER_SECS}s for master rank server to bind & ranks 0,1 to register..."
sleep "$WAIT_AFTER_MASTER_SECS"

if ! kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "error: master srun (pid=${MASTER_PID}) exited during head-start window; see master log." >&2
    say "===== launcher early-fail: master died during head-start ====="
    wait "$MASTER_PID" || true
    exit 1
fi

# ---------------------------------------------------------------------------
# 9. Start the other 2 nodes (they connect to the master's rank server).
# ---------------------------------------------------------------------------
declare -a WORKER_PIDS=()
for w in "${WORKER_NODES[@]}"; do
    pid=$(run_elastic_on "$w" "worker_${w}.log" --tcp-server "$MASTER_IP" "${EXTRA_ARGS[@]}")
    WORKER_PIDS+=("$pid")
    say "worker ${w} pid (background srun) = ${pid}"
done

# ---------------------------------------------------------------------------
# 10. Wait for everyone.
# ---------------------------------------------------------------------------
final_rc=0
declare -A NODE_RC=()

say "waiting for master ${MASTER_NODE} (pid=${MASTER_PID})..."
if wait "$MASTER_PID"; then
    NODE_RC["$MASTER_NODE"]=0
    say "master ${MASTER_NODE} exited rc=0"
else
    rc=$?
    NODE_RC["$MASTER_NODE"]="$rc"
    say "master ${MASTER_NODE} exited rc=${rc}"
    final_rc=$rc
fi
for i in "${!WORKER_NODES[@]}"; do
    w="${WORKER_NODES[$i]}"
    wpid="${WORKER_PIDS[$i]}"
    say "waiting for worker ${w} (pid=${wpid})..."
    if wait "$wpid"; then
        NODE_RC["$w"]=0
        say "worker ${w} exited rc=0"
    else
        rc=$?
        NODE_RC["$w"]="$rc"
        say "worker ${w} exited rc=${rc}"
        if (( final_rc == 0 )); then final_rc=$rc; fi
    fi
done

# ---------------------------------------------------------------------------
# 11. Per-node cleanup report.
# ---------------------------------------------------------------------------
say "settling ${SETTLE_SECONDS}s before cleanup checks..."
sleep "$SETTLE_SECONDS"

CLEANUP_LOG="$RESULTS_DIR_HOST/cleanup_report.log"
{
    echo "===== CLEANUP REPORT $(ts) ====="
    for n in "${NODES[@]}"; do
        echo "--- node=${n} (elastic rc=${NODE_RC[$n]:-?}) ---"
        srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$n" \
            --container-image="$IMAGE" \
            --container-mounts="$MOUNT_SPEC" \
            --container-workdir="$CTR_NIXL" \
            bash -lc '
                set +e
                echo "hostname=$(hostname -s)"
                lp=$({ pgrep -af "elastic\.py|rank_server|spawn_main|torch.multiprocessing" 2>/dev/null || true; } | wc -l)
                echo "leftover_procs=${lp}"
                shm=$({ ls /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null || true; } | wc -l)
                echo "shm_torch_files=${shm}"
                if command -v ss >/dev/null 2>&1; then
                    ports=$({ ss -tlnH 2>/dev/null || true; } | awk "\$4 ~ /:(9999|10000)$/" | wc -l)
                else
                    ports=$({ netstat -tln 2>/dev/null || true; } | awk "\$4 ~ /:(9999|10000)$/" | wc -l)
                fi
                echo "ports_9999_10000=${ports}"
                gmem=$({ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true; } | awk "BEGIN{s=0} {s+=\$1} END{print s+0}")
                echo "gpu_mem_used_mib=${gmem}"
                apps=$({ nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true; } | awk "NF>0" | wc -l)
                echo "nvsmi_compute_apps=${apps}"
            ' 2>&1 | sed "s/^/[cleanup:${n}] /"
    done
    echo "===== end CLEANUP REPORT ====="
} | tee -a "$CLEANUP_LOG"

# ---------------------------------------------------------------------------
# 12. Final summary
# ---------------------------------------------------------------------------
say "===== launcher done rc=${final_rc} ====="
say "per-node rc:"
for n in "${NODES[@]}"; do
    say "  ${n}: ${NODE_RC[$n]:-?}"
done
say "logs:"
say "  ${LAUNCHER_LOG}"
say "  ${RESULTS_DIR_HOST}/master_${MASTER_NODE}.log"
for w in "${WORKER_NODES[@]}"; do
    say "  ${RESULTS_DIR_HOST}/worker_${w}.log"
done
say "  ${CLEANUP_LOG}"

exit "$final_rc"
