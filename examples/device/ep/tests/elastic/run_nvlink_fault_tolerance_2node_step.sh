#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-node body for the two-node NVLink fault-tolerance test.
#
# This script is invoked once per node inside a single
#   `srun -N2 --ntasks-per-node=1 --container-image=... bash step.sh`
# step. SLURM sets SLURM_NODEID=0 on master and SLURM_NODEID=1 on worker.
# Both tasks run this same body; the only behavioural difference is that the
# worker appends `--tcp-server <master_hostname>` to its elastic.py args.
#
# Symmetry with the 1-node script (run_nvlink_fault_tolerance_test.sh) is
# intentional: this script runs the same five cleanup probes locally before
# and after elastic.py, and emits the same CLEANUP REPORT block format. The
# only addition is a `[rank=N host=H role=R]` tag in the report header so
# the login-side sweep can split master vs worker.
#
# Required env (set by run_nvlink_fault_tolerance_2node_sweep.sh):
#   RUN_DIR          absolute container-side path (/workspace/lishapira/...)
#   TIMING           label, e.g. "before-dispatch", "dispatch-send-during-kernel"
#   ITER             iteration number, e.g. "1"
#   PLAN_FILE        e.g. "nvlink_fault_tolerance_2node.json"
#
# Optional env:
#   NUM_PROCESSES         workers per node (default 4)
#   FAULT_KILL_SIGNAL     elastic.py signal (default sigkill)
#   SETTLE_SECONDS        post-elastic settle wait (default 5)
#   MEM_LEAK_MIB          gpu mem delta threshold MiB (default 64)
#   FAULT_EVIDENCE_DIR    forwarded to elastic.py for in-kernel timings
#
# Args after `--` (or any positional args): forwarded to elastic.py verbatim,
# e.g. `--fault-kill-timing dispatch-send-during-kernel
# --in-kernel-fault-spin-cycles 100000000`.

set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Required env
# ---------------------------------------------------------------------------
: "${RUN_DIR:?RUN_DIR not set (login-side sweep should pass this via --export)}"
: "${TIMING:?TIMING not set}"
: "${ITER:?ITER not set}"
: "${PLAN_FILE:?PLAN_FILE not set}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
FAULT_KILL_SIGNAL="${FAULT_KILL_SIGNAL:-sigkill}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
MEM_LEAK_MIB="${MEM_LEAK_MIB:-64}"

# ---------------------------------------------------------------------------
# 1. Source the per-clone bring-up script. Defaults to this clone's
#    setup_node.sh (sourced from /workspace/dyogev/nixl-4165b16) so the
#    in-kernel-marker bindings built from THIS source are loaded; can be
#    overridden via FT_SETUP_NODE_SH if a different prebuilt tree is wanted.
#    Image bakes an older nixl_ep at /usr/local that shadows ours otherwise.
# ---------------------------------------------------------------------------
FT_SETUP_NODE_SH="${FT_SETUP_NODE_SH:-/workspace/dyogev/nixl-4165b16/setup_node.sh}"
# shellcheck disable=SC1091,SC1090
source "${FT_SETUP_NODE_SH}" >/dev/null 2>&1 || true
unset UCX_TLS

# ---------------------------------------------------------------------------
# 2. Identify our role within this run. With the davidyogev-style master-
#    first/worker-second split, the orchestrator runs two separate
#    `srun --nodes=1` jobs and SLURM_NODEID is 0 on BOTH of them; the
#    orchestrator therefore passes FT_NODEID=0 (master) or FT_NODEID=1
#    (worker) explicitly. Fall back to SLURM_NODEID when FT_NODEID isn't
#    set so the legacy single-srun-N2 path (and any 1-node call) still
#    works.
# ---------------------------------------------------------------------------
NODEID="${FT_NODEID:-${SLURM_NODEID:-0}}"
if (( NODEID == 0 )); then ROLE=master; else ROLE=worker; fi
SELF_HOST=$(hostname -s)

# Discover the master hostname for the worker's --tcp-server. SLURM_STEP_NODELIST
# is the step's nodelist in compact form (e.g. "lyris[0068,0071]").
resolve_master_host() {
    local list="${SLURM_STEP_NODELIST:-${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}}"
    [[ -z "${list}" ]] && { echo ""; return; }
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "${list}" 2>/dev/null | head -n 1
        return
    fi
    # Python fallback (container lacks slurm client tools).
    python3 - "${list}" <<'PY'
import sys, re
spec = sys.argv[1]
tokens, buf, depth = [], "", 0
for c in spec:
    if c == "[":
        depth += 1; buf += c
    elif c == "]":
        depth -= 1; buf += c
    elif c == "," and depth == 0:
        if buf:
            tokens.append(buf); buf = ""
    else:
        buf += c
if buf:
    tokens.append(buf)
out = []
for part in tokens:
    m = re.match(r"^(.+)\[(.+)\]$", part)
    if m:
        prefix, inner = m.group(1), m.group(2)
        for item in inner.split(","):
            item = item.strip()
            if "-" in item:
                a, b = item.split("-", 1)
                width = max(len(a), len(b))
                for i in range(int(a), int(b)+1):
                    out.append(f"{prefix}{i:0{width}d}")
            else:
                out.append(f"{prefix}{item}")
    else:
        out.append(part)
print(out[0] if out else "")
PY
}
MASTER_HOST="${MASTER_HOST:-$(resolve_master_host)}"
if [[ -z "${MASTER_HOST}" ]]; then
    echo "[step ${SELF_HOST}/${ROLE}] FATAL: cannot resolve master hostname from \
SLURM_STEP_NODELIST=${SLURM_STEP_NODELIST:-unset} \
SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 3. Per-rank output log. tee everything below so srun's combined output also
#    shows it in real time (useful when watching the sweep from the login
#    side), while the lustre file gives the sweep a clean per-rank source.
# ---------------------------------------------------------------------------
mkdir -p "${RUN_DIR}"
chmod 2777 "${RUN_DIR}" 2>/dev/null || true
RANK_LOG="${RUN_DIR}/${TIMING}__iter${ITER}__rank${NODEID}_${ROLE}_${SELF_HOST}.log"
# Pre-create the rank log so a permission failure surfaces NOW (in srun's
# combined log) rather than getting eaten by `tee` inside process substitution.
# `2>&1 1>&2` so the diagnostic lands on stderr (which propagates) regardless
# of whether stdout aggregation is healthy.
if ! : > "${RANK_LOG}" 2>/dev/null; then
    printf '[step %s/%s] WARN: cannot create rank log %s (perm/uid issue, will fall back to combined log)\n' \
        "${SELF_HOST}" "${ROLE}" "${RANK_LOG}" >&2
    RANK_LOG=/dev/null
fi
# shellcheck disable=SC2094
exec > >(tee "${RANK_LOG}") 2>&1

echo "[step ${SELF_HOST}/${ROLE}] start timing=${TIMING} iter=${ITER} plan=${PLAN_FILE} master=${MASTER_HOST}"

# ---------------------------------------------------------------------------
# 4. Cleanup probes -- byte-for-byte the same five helpers used by the
#    1-node launcher (run_nvlink_fault_tolerance_test.sh). Running them here,
#    inside the container on the actual compute node, is what gives the
#    2-node sweep the same coverage as the 1-node sweep.
# ---------------------------------------------------------------------------
gpu_mem_used_mib() {
    { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true; } \
        | awk 'BEGIN{s=0} {s+=$1} END{print s+0}'
}
leftover_proc_count() {
    { pgrep -af 'elastic\.py|rank_server|spawn_main|torch.multiprocessing' 2>/dev/null || true; } \
        | wc -l
}
shm_leak_count() {
    # shellcheck disable=SC2012
    { ls /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null || true; } | wc -l
}
ports_in_use() {
    local count=0
    if command -v ss >/dev/null 2>&1; then
        count=$({ ss -tlnH 2>/dev/null || true; } | awk '$4 ~ /:(9999|10000)$/' | wc -l)
    elif command -v netstat >/dev/null 2>&1; then
        count=$({ netstat -tln 2>/dev/null || true; } | awk '$4 ~ /:(9999|10000)$/' | wc -l)
    fi
    echo "${count:-0}"
}
nvsmi_compute_app_count() {
    { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true; } \
        | awk 'NF>0' | wc -l
}

PRE_GPU=$(gpu_mem_used_mib)
PRE_PROCS=$(leftover_proc_count)
PRE_SHM=$(shm_leak_count)
PRE_PORTS=$(ports_in_use)
PRE_APPS=$(nvsmi_compute_app_count)

# ---------------------------------------------------------------------------
# 5. Build elastic.py args. Worker connects to master via --tcp-server.
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --plan "${PLAN_FILE}"
    --num-processes "${NUM_PROCESSES}"
    --fault-kill-signal "${FAULT_KILL_SIGNAL}"
)
if [[ -n "${FAULT_EVIDENCE_DIR:-}" ]]; then
    COMMON_ARGS+=(--fault-evidence-dir "${FAULT_EVIDENCE_DIR}")
fi
if [[ "${ROLE}" == "worker" ]]; then
    COMMON_ARGS+=(--tcp-server "${MASTER_HOST}")
fi

FT_TEST_DIR="${FT_TEST_DIR:-/workspace/dyogev/nixl-4165b16/examples/device/ep/tests/elastic}"
cd "${FT_TEST_DIR}"
echo "[step ${SELF_HOST}/${ROLE}] cmd: python3 -u elastic.py ${COMMON_ARGS[*]} $* (cwd=${FT_TEST_DIR})"

# ---------------------------------------------------------------------------
# 6. Run elastic.py. SIGKILL on a child rank inside elastic.py is the normal
#    case for fault timings; elastic.py is responsible for noticing via the
#    mask and exiting 0. We capture the exit code rather than letting it
#    propagate through `set -e` so we can still emit a CLEANUP REPORT.
# ---------------------------------------------------------------------------
set +e
python3 -u elastic.py "${COMMON_ARGS[@]}" "$@"
TEST_RC=$?
set -e || true

# ---------------------------------------------------------------------------
# 7. Settle + post-snapshot. Settle gives UCX/driver time to release GPU
#    memory and unlink /dev/shm before we probe again.
# ---------------------------------------------------------------------------
sleep "${SETTLE_SECONDS}"
POST_GPU=$(gpu_mem_used_mib)
POST_PROCS=$(leftover_proc_count)
POST_SHM=$(shm_leak_count)
POST_PORTS=$(ports_in_use)
POST_APPS=$(nvsmi_compute_app_count)
GPU_DELTA=$(( POST_GPU - PRE_GPU ))

# ---------------------------------------------------------------------------
# 8. Build issues list (same five rules as the 1-node script).
# ---------------------------------------------------------------------------
issues=()
if (( POST_PROCS > PRE_PROCS )); then
    issues+=("leftover_procs:${PRE_PROCS}->${POST_PROCS}")
fi
if (( POST_SHM > PRE_SHM )); then
    issues+=("shm_leak:${PRE_SHM}->${POST_SHM}")
fi
if (( POST_PORTS > 0 )); then
    issues+=("ports_in_use:${POST_PORTS}")
fi
if (( POST_APPS > PRE_APPS )); then
    issues+=("gpu_compute_apps:${PRE_APPS}->${POST_APPS}")
fi
if (( GPU_DELTA > MEM_LEAK_MIB )); then
    issues+=("gpu_mem_delta:${GPU_DELTA}MiB>threshold:${MEM_LEAK_MIB}MiB")
fi

# ---------------------------------------------------------------------------
# 9. CLEANUP REPORT block. Keys are identical to the 1-node block so the
#    sweep's parser doesn't need a special case. The header line is tagged
#    with `[rank=N host=H role=R]` so the sweep can section the combined
#    log into master/worker halves and grade each independently.
# ---------------------------------------------------------------------------
printf '\n===================== CLEANUP REPORT [rank=%s host=%s role=%s] =====================\n' \
    "${NODEID}" "${SELF_HOST}" "${ROLE}"
printf 'settle_seconds=%s test_exit_code=%s\n' "${SETTLE_SECONDS}" "${TEST_RC}"
printf 'leftover_procs:        pre=%s post=%s\n' "${PRE_PROCS}" "${POST_PROCS}"
printf 'shm_torch_files:       pre=%s post=%s\n' "${PRE_SHM}" "${POST_SHM}"
printf 'ports(9999,10000):     post=%s\n' "${POST_PORTS}"
printf 'nvsmi_compute_apps:    pre=%s post=%s\n' "${PRE_APPS}" "${POST_APPS}"
printf 'gpu_mem_used_mib:      pre=%s post=%s delta=%s threshold=%s\n' \
    "${PRE_GPU}" "${POST_GPU}" "${GPU_DELTA}" "${MEM_LEAK_MIB}"
if (( ${#issues[@]} == 0 )); then
    printf 'cleanup_result:        CLEAN\n'
else
    printf 'cleanup_result:        DIRTY (%s)\n' "${issues[*]}"
fi
printf '==========================================================\n'

# Detail dump on dirty (matches 1-node).
if (( ${#issues[@]} > 0 )); then
    printf '\n--- detail[%s=%s]: leftover processes ---\n' "${ROLE}" "${SELF_HOST}"
    pgrep -af 'elastic\.py|rank_server|spawn_main|torch.multiprocessing' \
        || printf '(none)\n'
    printf '\n--- detail[%s=%s]: /dev/shm leaks ---\n' "${ROLE}" "${SELF_HOST}"
    ls -la /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null \
        || printf '(none)\n'
    printf '\n--- detail[%s=%s]: nvidia-smi compute apps ---\n' "${ROLE}" "${SELF_HOST}"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null \
        || printf '(none)\n'
fi

exit "${TEST_RC}"
