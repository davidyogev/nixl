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
# Maximum time to wait for the master to become ready (rank server bound +
# GPUS_PER_NODE workers registered). The launcher actively polls the master
# log; this is just the upper bound before we give up. Cold container start
# can take ~20s on a chilly cache; we allow generous headroom but exit as
# soon as ranks are registered, usually within 5-10s.
MASTER_READY_TIMEOUT_SECS=${MASTER_READY_TIMEOUT_SECS:-60}
MASTER_READY_POLL_INTERVAL_SECS=${MASTER_READY_POLL_INTERVAL_SECS:-1}
# Optional floor wait AFTER ranks register (e.g. set to 2 if you want to be
# extra-safe that the rank server has fully transitioned before letting
# worker nodes connect). Default 0: proceed immediately.
WAIT_AFTER_MASTER_READY_SECS=${WAIT_AFTER_MASTER_READY_SECS:-0}
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
# CRITICAL: write log lines to stderr, not stdout. The launcher uses
# `pid=$(run_elastic_on ...)` to capture the bg srun pid via `echo $!`; if
# say() wrote to stdout, every dispatch log line would also land in $pid and
# `wait`/`kill -0` would explode on multi-line garbage. stderr still ends up
# in the launcher log + terminal because of the outer `2>&1` after tee.
say() { echo "[$(ts)] [launcher] $*" >&2; }

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
say "master_ready_timeout_secs=${MASTER_READY_TIMEOUT_SECS} poll=${MASTER_READY_POLL_INTERVAL_SECS}s wait_after_master_ready_secs=${WAIT_AFTER_MASTER_READY_SECS}"
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
# 6. Resolve the master endpoint workers will use to reach the rank server.
#    Default: just use the SLURM-provided hostname ($MASTER_NODE). On a
#    SLURM/InfiniBand cluster sibling nodes can always DNS-resolve each
#    other's hostnames; this avoids the rabbit hole of picking the "right"
#    IP from `hostname -I` (which often returns a link-local 169.254.x.x or
#    a management interface as its first entry, neither of which workers
#    can reach).
#
#    If your cluster requires a specific fabric IP for the rank-server
#    traffic, override MASTER_IP=<addr> before invoking this script. Empty
#    string means "use $MASTER_NODE".
# ---------------------------------------------------------------------------
MASTER_ENDPOINT="${MASTER_IP:-$MASTER_NODE}"
say "master_endpoint=${MASTER_ENDPOINT}  (override with MASTER_IP=... env var)"

# ---------------------------------------------------------------------------
# 6b. EXIT trap: kill any bg srun jobs we spawned so they don't outlive the
#     launcher and hog ports / GPU memory for the next attempt. Without this
#     trap, an early-fail (e.g. EADDRINUSE on master) leaves the worker srun
#     bg jobs running and the next launcher run collides with them on
#     port 9999/10000.
# ---------------------------------------------------------------------------
MASTER_PID=""
declare -a WORKER_PIDS=()
cleanup_bg_srun() {
    local rc=$?
    local p
    for p in "$MASTER_PID" "${WORKER_PIDS[@]:-}"; do
        [[ -z "$p" ]] && continue
        if kill -0 "$p" 2>/dev/null; then
            say "trap: terminating bg srun pid=$p"
            kill -TERM "$p" 2>/dev/null || true
        fi
    done
    # Give srun a moment to propagate the signal to its job step.
    sleep 1
    for p in "$MASTER_PID" "${WORKER_PIDS[@]:-}"; do
        [[ -z "$p" ]] && continue
        if kill -0 "$p" 2>/dev/null; then
            kill -KILL "$p" 2>/dev/null || true
        fi
    done
    exit "$rc"
}
trap cleanup_bg_srun EXIT

# ---------------------------------------------------------------------------
# 6c. Pre-flight: make sure the master's rank-server ports aren't already
#     held by an orphan from a previous run. If they are, bail loudly --
#     the actual elastic.py would crash with EADDRINUSE several seconds
#     later, after a useless container start.
# ---------------------------------------------------------------------------
say "pre-flight: checking master ports 9999/10000 are free on ${MASTER_NODE}..."
port_check=$(srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$MASTER_NODE" \
    bash -c '{ ss -tln 2>/dev/null || netstat -tln 2>/dev/null || true; } | awk "\$4 ~ /:(9999|10000)\$/ {print \$4}"' 2>/dev/null || true)
if [[ -n "$port_check" ]]; then
    say "FATAL: rank-server ports already in use on ${MASTER_NODE}:"
    echo "$port_check" | sed "s/^/  /" >&2
    say "An orphan from a previous run is holding the ports. Cleanup:"
    say "  scancel \$SLURM_JOB_ID   # then re-allocate (-N 3 ... --pty bash)"
    say "(this also nukes your current --pty shell -- re-srun to come back.)"
    exit 1
fi
say "pre-flight: master ports clear"

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
MASTER_LOG="${RESULTS_DIR_HOST}/master_${MASTER_NODE}.log"
MASTER_PID=$(run_elastic_on "$MASTER_NODE" "master_${MASTER_NODE}.log" "${EXTRA_ARGS[@]}")
say "master pid (background srun) = ${MASTER_PID}"

# Wait until the master is READY = GPUS_PER_NODE workers have registered
# with the rank server. We poll the master log for lines like:
#     "Process <torch_pid> -> global_rank=<n>, local_rank=<m>"
# Once we see GPUS_PER_NODE such lines, ranks 0..GPUS_PER_NODE-1 are pinned
# to the master and it's safe to launch the worker nodes. This is
# dramatically faster than a flat sleep (~5-10s hot, vs. having to assume
# a worst-case 25s+ cold start), AND it can't proceed early when the
# cluster is slow.
say "polling master log for ${GPUS_PER_NODE} rank registrations (timeout=${MASTER_READY_TIMEOUT_SECS}s)..."
deadline=$(( $(date +%s) + MASTER_READY_TIMEOUT_SECS ))
master_ready=0
while (( $(date +%s) < deadline )); do
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        say "===== launcher early-fail: master died before becoming ready ====="
        say "tail of master log (${MASTER_LOG}):"
        tail -n 50 "$MASTER_LOG" 2>&1 | sed "s/^/  /" >&2
        wait "$MASTER_PID" 2>/dev/null || true
        exit 1
    fi
    if [[ -f "$MASTER_LOG" ]]; then
        n_registered=$(grep -c '^Process [0-9]\+ -> global_rank=' "$MASTER_LOG" 2>/dev/null || echo 0)
        if (( n_registered >= GPUS_PER_NODE )); then
            elapsed=$(( $(date +%s) - (deadline - MASTER_READY_TIMEOUT_SECS) ))
            say "master ready: ${n_registered}/${GPUS_PER_NODE} ranks registered after ${elapsed}s"
            master_ready=1
            break
        fi
    fi
    sleep "$MASTER_READY_POLL_INTERVAL_SECS"
done
if (( master_ready == 0 )); then
    say "===== launcher early-fail: master did not register ${GPUS_PER_NODE} ranks within ${MASTER_READY_TIMEOUT_SECS}s ====="
    say "tail of master log (${MASTER_LOG}):"
    tail -n 50 "$MASTER_LOG" 2>&1 | sed "s/^/  /" >&2
    exit 1
fi

if (( WAIT_AFTER_MASTER_READY_SECS > 0 )); then
    say "optional floor wait: sleeping ${WAIT_AFTER_MASTER_READY_SECS}s after master ready..."
    sleep "$WAIT_AFTER_MASTER_READY_SECS"
fi

# ---------------------------------------------------------------------------
# 9. Start the other 2 nodes (they connect to the master's rank server).
# ---------------------------------------------------------------------------
# WORKER_PIDS was pre-declared near the EXIT trap so the trap can see it
# even if we early-fail before this loop runs.
for w in "${WORKER_NODES[@]}"; do
    pid=$(run_elastic_on "$w" "worker_${w}.log" --tcp-server "$MASTER_ENDPOINT" "${EXTRA_ARGS[@]}")
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
