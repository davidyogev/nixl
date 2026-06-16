#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two-node NVLink fault-tolerance sweep.
#
# Runs once per fault-kill-timing across two compute nodes. Each iteration is
# a single `srun -N2 --ntasks-per-node=1 --container-image=...` step that
# invokes run_nvlink_fault_tolerance_2node_step.sh on each node. That step
# script identifies itself via SLURM_NODEID (0=master, 1=worker), runs the
# same 5 cleanup probes locally as the 1-node launcher does, runs
# elastic.py, and emits a CLEANUP REPORT to a per-rank log file under
# $RUN_DIR. After srun returns this sweep aggregates the two per-rank logs
# into a combined per-timing log and verdicts it the same way
# run_nvlink_fault_tolerance_sweep.sh does for the 1-node case.
#
# Design (mirrors the 1-node sweep almost 1:1):
#   * Orchestrator stays on the LOGIN node where srun is available outside
#     the container -- the container image ships without slurm client tools.
#   * Each timing's elastic.py run is ONE srun step with TWO tasks (one per
#     node). SLURM starts both tasks concurrently; TCPStore handles the
#     master<->worker rendezvous. There is no nested srun, no manual
#     sleep-and-pray, no host->container path shimming.
#   * Cleanup probes run LOCALLY inside the container on each node, exactly
#     as the 1-node launcher does. Per-rank logs let the sweep grade master
#     and worker independently and label any DIRTY result by node.
#
# Required env (set by the user's salloc or shell):
#   SLURM_JOB_ID           the active job allocation
#   SLURM_JOB_NODELIST     two-node nodelist (or SLURM_NODELIST)
#   CONTAINER_IMAGE        path to nixl-hybrid-ep-cuda2.sqsh
#   CONTAINER_MOUNTS       e.g. /lustre/.../lishapira:/workspace/lishapira
#
# Optional env:
#   SPIN_CYCLES            in-kernel marker spin (default 100000000)
#   ITERATIONS             iterations per timing (default 1)
#   VICTIM_PLAN            default nvlink_fault_tolerance_2node.json
#   BASELINE_PLAN          default nvlink_fault_tolerance_2node_baseline.json
#   RUN_DIR                output dir (default results/2node_<utc>_job<id>_<nodes>)
#   TIMINGS                space-separated subset to run (default = all 12)
#   MEM_LEAK_MIB           GPU memory delta tolerance per node (default 64)
#   SETTLE_SECONDS         post-run settle wait (default 5)
#   NUM_PROCESSES          workers per node (default 4)
#   FAULT_KILL_SIGNAL      elastic.py signal (default sigkill)

set -uo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "${TEST_DIR}"

# ---------------------------------------------------------------------------
# Path mapping host <-> container (lustre is bind-mounted to /workspace/...).
# ---------------------------------------------------------------------------
LUSTRE_HOST_PREFIX="${LUSTRE_HOST_PREFIX:-/lustre/fsw/network_research_advdev/lishapira}"
LUSTRE_CONTAINER_PREFIX="${LUSTRE_CONTAINER_PREFIX:-/workspace/lishapira}"
host_to_container_path() {
    local p="$1"
    printf '%s' "${p/#${LUSTRE_HOST_PREFIX}/${LUSTRE_CONTAINER_PREFIX}}"
}

STEP_SCRIPT_HOST="${TEST_DIR}/run_nvlink_fault_tolerance_2node_step.sh"
STEP_SCRIPT_CONTAINER=$(host_to_container_path "${STEP_SCRIPT_HOST}")
[[ -f "${STEP_SCRIPT_HOST}" ]] || { echo "missing step script: ${STEP_SCRIPT_HOST}" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Required SLURM + container env
# ---------------------------------------------------------------------------
: "${SLURM_JOB_ID:?SLURM_JOB_ID not set; run from inside an active salloc}"
SLURM_LIST="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
: "${SLURM_LIST:?SLURM_JOB_NODELIST/SLURM_NODELIST not set}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE not set (path to nixl-hybrid-ep-cuda2.sqsh)}"
: "${CONTAINER_MOUNTS:?CONTAINER_MOUNTS not set (e.g. /lustre/.../lishapira:/workspace/lishapira)}"

SPIN_CYCLES="${SPIN_CYCLES:-100000000}"
ITERATIONS="${ITERATIONS:-1}"
VICTIM_PLAN="${VICTIM_PLAN:-nvlink_fault_tolerance_2node.json}"
BASELINE_PLAN="${BASELINE_PLAN:-nvlink_fault_tolerance_2node_baseline.json}"
MEM_LEAK_MIB="${MEM_LEAK_MIB:-64}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
FAULT_KILL_SIGNAL="${FAULT_KILL_SIGNAL:-sigkill}"

# Per-clone setup script + test dir paths (container-side). step.sh sources
# the setup script and cds into the test dir before running elastic.py; the
# build-sanity probe sources the same setup script. Defaults match this
# clone's location at /workspace/dyogev/nixl-4165b16.
export FT_SETUP_NODE_SH="${FT_SETUP_NODE_SH:-/workspace/dyogev/nixl-4165b16/setup_node.sh}"
export FT_TEST_DIR="${FT_TEST_DIR:-/workspace/dyogev/nixl-4165b16/examples/device/ep/tests/elastic}"

DEFAULT_TIMINGS=(
    # CPU-level kills
    before-dispatch
    after-dispatch
    between-dispatch-combine
    before-combine
    after-combine
    dispatch-between-send-receive
    combine-between-send-receive
    # In-kernel kills: 4 phases x 2 hook modes = 8. `-no-hook` = fused
    # variant (return_recv_hook=False); `-hook-separated` = split variant
    # (return_recv_hook=True, send/recv split with a host hook between).
    dispatch-send-during-kernel-no-hook
    dispatch-send-during-kernel-hook-separated
    dispatch-receive-during-kernel-no-hook
    dispatch-receive-during-kernel-hook-separated
    combine-send-during-kernel-no-hook
    combine-send-during-kernel-hook-separated
    combine-receive-during-kernel-no-hook
    combine-receive-during-kernel-hook-separated
)
if [[ -n "${TIMINGS:-}" ]]; then
    # shellcheck disable=SC2206
    TIMINGS_ARR=( ${TIMINGS} )
else
    TIMINGS_ARR=( "${DEFAULT_TIMINGS[@]}" )
fi

# `-no-hook` and `-hook-separated` variants of the same (op, phase) cell
# target the same internal kernel phase (target id), so both entries of
# each pair share the same value.
declare -A EXPECTED_TARGET=(
    [dispatch-send-during-kernel-hook-separated]=1
    [dispatch-send-during-kernel-no-hook]=1
    [dispatch-receive-during-kernel-hook-separated]=2
    [dispatch-receive-during-kernel-no-hook]=2
    [combine-send-during-kernel-hook-separated]=3
    [combine-send-during-kernel-no-hook]=3
    [combine-receive-during-kernel-hook-separated]=4
    [combine-receive-during-kernel-no-hook]=4
)

# ---------------------------------------------------------------------------
# Resolve nodelist for the run-dir name and for the master hostname inside
# step.sh (passed via env so it doesn't have to parse the nodelist itself).
# ---------------------------------------------------------------------------
parse_nodelist() {
    local list="$1"
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "${list}"
        return
    fi
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
print("\n".join(out))
PY
}

mapfile -t NODES < <(parse_nodelist "${SLURM_LIST}")
if (( ${#NODES[@]} < 2 )); then
    echo "error: need at least 2 nodes, got ${#NODES[@]}: ${NODES[*]:-none}" >&2
    exit 2
fi
MASTER_HOST="${NODES[0]}"
WORKER_HOST="${NODES[1]}"
NODES_STR=$(IFS=_; echo "${NODES[*]}")

# ---------------------------------------------------------------------------
# Output dirs (host paths -- mkdir on lustre; container paths -- passed to
# step.sh via --export).
# ---------------------------------------------------------------------------
UTC=$(date -u +%Y%m%d_%H%M%S)
JOB_TAG="${SLURM_JOB_ID}"
RUN_DIR_HOST="${RUN_DIR:-${TEST_DIR}/results/2node_${UTC}_job${JOB_TAG}_${NODES_STR}}"
RUN_DIR_CONTAINER=$(host_to_container_path "${RUN_DIR_HOST}")
mkdir -p "${RUN_DIR_HOST}"
# Open perms so the in-container step.sh (running as the launching user, who
# may not be in this tree's primary group) can create per-rank log files via
# tee. mkdir defaults to ~0755 + setgid, which leaves "others" with no write
# even when the parent is 2777, so we explicitly chmod each created dir.
chmod 2777 "${RUN_DIR_HOST}" 2>/dev/null || true
EVIDENCE_DIR_HOST="${RUN_DIR_HOST}/evidence"
EVIDENCE_DIR_CONTAINER=$(host_to_container_path "${EVIDENCE_DIR_HOST}")
mkdir -p "${EVIDENCE_DIR_HOST}"
chmod 2777 "${EVIDENCE_DIR_HOST}" 2>/dev/null || true
SUMMARY="${RUN_DIR_HOST}/SUMMARY.md"

log_summary() { printf '%s\n' "$*" >> "${SUMMARY}"; }

# ---------------------------------------------------------------------------
# Pre-flight: verify the lustre install has the in-kernel marker symbols.
# Single 1-task srun on master inside the container. Matches the 1-node
# sweep's verify_build, but executes from outside the container via srun.
# ---------------------------------------------------------------------------
verify_build() {
    local probe_log="${RUN_DIR_HOST}/build_probe.log"
    local setup_sh="${FT_SETUP_NODE_SH:-/workspace/dyogev/nixl-4165b16/setup_node.sh}"
    local probe_py='source '"${setup_sh}"' >/dev/null 2>&1
python3 - <<'\''PY'\''
import nixl_ep
needed = (
    "enable_in_kernel_fault_marker",
    "disable_in_kernel_fault_marker",
    "get_in_kernel_fault_marker_snapshot",
    "query_mask_buffer",
)
missing = [m for m in needed if m not in dir(nixl_ep.Buffer)]
if missing:
    raise SystemExit("missing Buffer methods: " + ", ".join(missing))
print("Buffer methods OK")
PY'
    local out
    out=$(srun --jobid="${SLURM_JOB_ID}" --overlap \
        --nodes=1 --ntasks=1 --nodelist="${MASTER_HOST}" \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        --container-workdir=/workspace/dyogev \
        bash -c "${probe_py}" 2>&1)
    {
        echo "=== build probe (srun on ${MASTER_HOST}, in container) ==="
        echo "${out}"
    } > "${probe_log}"
    if ! grep -q "Buffer methods OK" "${probe_log}"; then
        echo "BUILD SANITY FAILED -- rebuild from nvlink_fault_tolerance branch." >&2
        echo "See ${probe_log}" >&2
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Master-ready polling knobs (port of davidyogev's nvlink-fault-tolerance-3node
# launcher commits e315aa3 "poll for ready + clean up bg srun" and 34a5535
# "fix wait/poll/cleanup bugs surfaced on Lyris"). The previous single
# `srun -N2 --ntasks-per-node=1` had master and worker race for rank-server
# binding; on Lyris the master sometimes hadn't bound port 10000 by the time
# the worker tried to connect, the master's own children failed at
# `RankClient.get_rank()`, and `torch.multiprocessing` swallowed the
# traceback into "Worker processes failed: ... exit code 1".
# ---------------------------------------------------------------------------
MASTER_READY_TIMEOUT_SECS="${MASTER_READY_TIMEOUT_SECS:-60}"
MASTER_READY_POLL_INTERVAL_SECS="${MASTER_READY_POLL_INTERVAL_SECS:-1}"
WAIT_AFTER_MASTER_READY_SECS="${WAIT_AFTER_MASTER_READY_SECS:-0}"

# ---------------------------------------------------------------------------
# Per-iteration EXIT trap state. We launch master + worker as background
# srun jobs; if the orchestrator dies (^C, error in caller) we don't want
# them outliving us and holding ports 9999/10000 for the next iteration.
# Updated by run_one before/after each launch; cleanup_bg_srun walks both.
# ---------------------------------------------------------------------------
RUN_ONE_MASTER_PID=""
RUN_ONE_WORKER_PID=""
cleanup_bg_srun() {
    local rc=$?
    local p
    for p in "${RUN_ONE_MASTER_PID}" "${RUN_ONE_WORKER_PID}"; do
        [[ -z "${p}" ]] && continue
        if kill -0 "${p}" 2>/dev/null; then
            echo "[2node-sweep] trap: terminating bg srun pid=${p}" >&2
            kill -TERM "${p}" 2>/dev/null || true
        fi
    done
    sleep 1
    for p in "${RUN_ONE_MASTER_PID}" "${RUN_ONE_WORKER_PID}"; do
        [[ -z "${p}" ]] && continue
        if kill -0 "${p}" 2>/dev/null; then
            kill -KILL "${p}" 2>/dev/null || true
        fi
    done
    return "${rc}"
}
trap 'cleanup_bg_srun; exit' INT TERM
trap 'cleanup_bg_srun' EXIT

# ---------------------------------------------------------------------------
# preflight_master_ports: srun a quick `ss -tln` on the master and bail if
# ports 9999 (TCPStore) or 10000 (rank server) are still held by an orphan
# from a previous iteration. Without this, elastic.py would crash with
# EADDRINUSE inside a freshly-started container ~5s into the run --
# expensive way to learn we have a leak.
# ---------------------------------------------------------------------------
preflight_master_ports() {
    local hits
    hits=$(srun --jobid="${SLURM_JOB_ID}" --overlap \
        --nodes=1 --ntasks=1 --nodelist="${MASTER_HOST}" \
        bash -c '{ ss -tln 2>/dev/null || netstat -tln 2>/dev/null || true; } | awk "\$4 ~ /:(9999|10000)\$/ {print \$4}"' \
        2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        echo "[2node-sweep] preflight FAIL: master ${MASTER_HOST} already holds these ports:" >&2
        echo "${hits}" | sed "s/^/  /" >&2
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# launch_one_node: dispatch a single-node srun in background, capture the
# bg pid via the global LAUNCH_BG_PID side effect (NOT $() -- $() runs in a
# subshell; the bg srun reparents to init the moment the subshell exits and
# the parent can no longer `wait` on it -> instant rc=127 even though
# elastic.py is running fine. davidyogev's commit 34a5535 hit and fixed
# this exact bug on Lyris).
# ---------------------------------------------------------------------------
LAUNCH_BG_PID=""
launch_one_node() {
    local node="$1"; shift
    local log_file="$1"; shift
    local role_env="$1"; shift   # "FT_NODEID=0" or "FT_NODEID=1"
    local extra_export="$1"; shift  # comma-prefixed extras or empty
    local timing="$1"; shift
    local iter="$1"; shift
    local plan="$1"; shift

    srun --jobid="${SLURM_JOB_ID}" --overlap \
        --nodes=1 --ntasks=1 --nodelist="${node}" \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        --container-workdir=/workspace/dyogev \
        --export=ALL,RUN_DIR="${RUN_DIR_CONTAINER}",TIMING="${timing}",ITER="${iter}",PLAN_FILE="${plan}",FAULT_EVIDENCE_DIR="${EVIDENCE_DIR_CONTAINER}",NUM_PROCESSES="${NUM_PROCESSES}",FAULT_KILL_SIGNAL="${FAULT_KILL_SIGNAL}",SETTLE_SECONDS="${SETTLE_SECONDS}",MEM_LEAK_MIB="${MEM_LEAK_MIB}",MASTER_HOST="${MASTER_HOST}",FT_SETUP_NODE_SH="${FT_SETUP_NODE_SH}",FT_TEST_DIR="${FT_TEST_DIR}",${role_env}${extra_export} \
        bash "${STEP_SCRIPT_CONTAINER}" "$@" \
        > "${log_file}" 2>&1 &
    LAUNCH_BG_PID=$!
}

# ---------------------------------------------------------------------------
# run_one: launch master first, poll for readiness (NUM_PROCESSES rank
# registrations in master log), then launch worker. Wait for both to
# finish, concatenate per-node logs into the combined out_log so the
# orchestrator's parser sees the same byte stream it always has.
#
#   $1 = plan filename (victim or baseline)
#   $2 = output combined log path (host side; concatenation of master+worker)
#   $3 = timing label
#   $4 = iteration number
#   rest = passed verbatim to elastic.py (e.g. --fault-kill-timing ...)
# ---------------------------------------------------------------------------
run_one() {
    local plan="$1"; shift
    local out_log="$1"; shift
    local timing="$1"; shift
    local iter="$1"; shift

    local master_log="${out_log}.master"
    local worker_log="${out_log}.worker"
    : > "${master_log}"
    : > "${worker_log}"

    # Pre-flight: ports 9999/10000 free on master.
    if ! preflight_master_ports; then
        echo "[2node-sweep] aborting iteration; cleanup with: scancel ${SLURM_JOB_ID}" >&2
        cat "${master_log}" "${worker_log}" > "${out_log}" 2>/dev/null || true
        return 2
    fi

    # ---- Phase 1: launch master, poll until ready --------------------------
    launch_one_node "${MASTER_HOST}" "${master_log}" "FT_NODEID=0" "" \
        "${timing}" "${iter}" "${plan}" "$@"
    RUN_ONE_MASTER_PID="${LAUNCH_BG_PID}"

    local deadline=$(( $(date +%s) + MASTER_READY_TIMEOUT_SECS ))
    local master_ready=0
    local n_registered=0
    while (( $(date +%s) < deadline )); do
        if ! kill -0 "${RUN_ONE_MASTER_PID}" 2>/dev/null; then
            echo "[2node-sweep] master ${MASTER_HOST} died during head-start; tail of ${master_log}:" >&2
            tail -n 30 "${master_log}" 2>&1 | sed "s/^/  /" >&2
            wait "${RUN_ONE_MASTER_PID}" 2>/dev/null || true
            local mrc=$?
            cat "${master_log}" "${worker_log}" > "${out_log}" 2>/dev/null || true
            RUN_ONE_MASTER_PID=""
            return "${mrc}"
        fi
        if [[ -f "${master_log}" ]]; then
            # `grep -c ... || true; ${var:-0}` defends against grep's
            # exit-1-on-no-match (davidyogev commit 34a5535 fixed the
            # `|| echo 0` form which produced "0\n0" and crashed (( ... )).
            n_registered=$(grep -c '^Process [0-9]\+ -> global_rank=' "${master_log}" 2>/dev/null || true)
            n_registered=${n_registered:-0}
            if (( n_registered >= NUM_PROCESSES )); then
                master_ready=1
                break
            fi
        fi
        sleep "${MASTER_READY_POLL_INTERVAL_SECS}"
    done

    if (( master_ready == 0 )); then
        echo "[2node-sweep] master not ready (${n_registered}/${NUM_PROCESSES} ranks) after ${MASTER_READY_TIMEOUT_SECS}s; tail of ${master_log}:" >&2
        tail -n 30 "${master_log}" 2>&1 | sed "s/^/  /" >&2
        kill -TERM "${RUN_ONE_MASTER_PID}" 2>/dev/null || true
        wait "${RUN_ONE_MASTER_PID}" 2>/dev/null || true
        cat "${master_log}" "${worker_log}" > "${out_log}" 2>/dev/null || true
        RUN_ONE_MASTER_PID=""
        return 1
    fi

    if (( WAIT_AFTER_MASTER_READY_SECS > 0 )); then
        sleep "${WAIT_AFTER_MASTER_READY_SECS}"
    fi

    # ---- Phase 2: launch worker --------------------------------------------
    launch_one_node "${WORKER_HOST}" "${worker_log}" "FT_NODEID=1" "" \
        "${timing}" "${iter}" "${plan}" "$@"
    RUN_ONE_WORKER_PID="${LAUNCH_BG_PID}"

    # ---- Wait for both. We collect both rcs but return the worse of the two.
    wait "${RUN_ONE_WORKER_PID}"
    local worker_rc=$?
    RUN_ONE_WORKER_PID=""
    wait "${RUN_ONE_MASTER_PID}"
    local master_rc=$?
    RUN_ONE_MASTER_PID=""

    # Concatenate per-node logs into the combined log the parser expects.
    # Master first so its CLEANUP REPORT [rank=0 ...] precedes worker's;
    # fault-evidence files go to ${EVIDENCE_DIR_CONTAINER} regardless.
    cat "${master_log}" "${worker_log}" > "${out_log}"
    rm -f "${master_log}" "${worker_log}"

    if (( master_rc != 0 )); then
        return "${master_rc}"
    fi
    return "${worker_rc}"
}

# ---------------------------------------------------------------------------
# expected_survivors_for_plan: ranks present (>=0) in the last plan phase.
# ---------------------------------------------------------------------------
expected_survivors_for_plan() {
    local plan="$1"
    python3 - "$plan" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
last = plan[-1]
print(sum(1 for r in last if r >= 0))
PY
}

# ---------------------------------------------------------------------------
# parse_cleanup_from_log: read one per-rank log, set CLEANUP_STATUS/_DETAIL.
# ---------------------------------------------------------------------------
parse_cleanup_from_log() {
    local log="$1"
    CLEANUP_STATUS="UNKNOWN"
    CLEANUP_DETAIL="no CLEANUP REPORT"
    [[ -f "${log}" ]] || { CLEANUP_DETAIL="missing log: ${log}"; return; }
    local result_line
    result_line=$(grep -E '^cleanup_result:' "${log}" 2>/dev/null | head -n 1 || true)
    [[ -z "${result_line}" ]] && return
    local delta_line delta
    delta_line=$(grep -E '^gpu_mem_used_mib:' "${log}" 2>/dev/null | head -n 1 || true)
    delta=$(printf '%s' "${delta_line}" | sed -n 's/.*delta=\(-\?[0-9]\+\).*/\1/p')
    if [[ "${result_line}" == *CLEAN* ]]; then
        CLEANUP_STATUS="PASS"
        CLEANUP_DETAIL=$(printf 'delta=%+dMiB' "${delta:-0}")
    else
        CLEANUP_STATUS="FAIL"
        local detail
        detail=$(printf '%s' "${result_line}" \
            | sed -n 's/^cleanup_result:[[:space:]]*DIRTY[[:space:]]*(\(.*\))$/\1/p')
        CLEANUP_DETAIL="${detail:-DIRTY}"
    fi
}

# ---------------------------------------------------------------------------
# rank_log_path: glob (with hostname embedded) the per-rank log path written
# by step.sh. Sets RANK_LOG to "" if missing.
# ---------------------------------------------------------------------------
rank_log_path() {
    local timing="$1"
    local iter="$2"
    local rank="$3"
    local role
    if [[ "${rank}" == "0" ]]; then role=master; else role=worker; fi
    local glob="${RUN_DIR_HOST}/${timing}__iter${iter}__rank${rank}_${role}_"*.log
    # shellcheck disable=SC2206
    local matches=( ${glob} )
    if [[ -e "${matches[0]}" ]]; then
        RANK_LOG="${matches[0]}"
    else
        RANK_LOG=""
    fi
}

# ---------------------------------------------------------------------------
# parse_cleanup_from_combined_log: extract the per-rank CLEANUP REPORT block
# out of the combined srun log. Each block starts with a header line of the
# form:
#   ===================== CLEANUP REPORT [rank=N host=H role=R] =====================
# and ends with a row of '=' characters. Sets CLEANUP_STATUS/DETAIL just like
# parse_cleanup_from_log. Used as a fallback when per-rank logs are missing.
# ---------------------------------------------------------------------------
parse_cleanup_from_combined_log() {
    local combined="$1"
    local rank="$2"
    CLEANUP_STATUS="UNKNOWN"
    CLEANUP_DETAIL="no CLEANUP REPORT (combined-log fallback)"
    [[ -f "${combined}" ]] || { CLEANUP_DETAIL="missing combined log: ${combined}"; return; }
    local block
    block=$(awk -v rank="${rank}" '
        $0 ~ "^=+ CLEANUP REPORT \\[rank="rank" "  { in_block = 1; print; next }
        in_block && /^=+$/                        { print; exit }
        in_block                                  { print }
    ' "${combined}" 2>/dev/null)
    [[ -z "${block}" ]] && { CLEANUP_DETAIL="no CLEANUP REPORT for rank=${rank} in ${combined}"; return; }
    local result_line
    result_line=$(printf '%s\n' "${block}" | grep -E '^cleanup_result:' | head -n 1 || true)
    [[ -z "${result_line}" ]] && return
    local delta_line delta
    delta_line=$(printf '%s\n' "${block}" | grep -E '^gpu_mem_used_mib:' | head -n 1 || true)
    delta=$(printf '%s' "${delta_line}" | sed -n 's/.*delta=\(-\?[0-9]\+\).*/\1/p')
    if [[ "${result_line}" == *CLEAN* ]]; then
        CLEANUP_STATUS="PASS"
        CLEANUP_DETAIL=$(printf 'delta=%+dMiB(combined)' "${delta:-0}")
    else
        CLEANUP_STATUS="FAIL"
        local detail
        detail=$(printf '%s' "${result_line}" \
            | sed -n 's/^cleanup_result:[[:space:]]*DIRTY[[:space:]]*(\(.*\))$/\1/p')
        CLEANUP_DETAIL="${detail:-DIRTY}(combined)"
    fi
}

# ---------------------------------------------------------------------------
# parse_cleanup_combined: PASS iff both ranks are CLEAN; FAIL if either is
# DIRTY; UNKNOWN if either log is missing. Detail string identifies which
# node failed. If a per-rank log is missing we fall back to scraping the
# combined srun log for that rank's CLEANUP REPORT block, so a healthy run
# isn't false-flagged just because tee couldn't create the per-rank file.
# ---------------------------------------------------------------------------
parse_cleanup_combined() {
    local timing="$1"
    local iter="$2"
    local m_status m_detail w_status w_detail
    local combined_log="${RUN_DIR_HOST}/${timing}__iter${iter}.log"
    rank_log_path "${timing}" "${iter}" 0; local m_log="${RANK_LOG}"
    rank_log_path "${timing}" "${iter}" 1; local w_log="${RANK_LOG}"
    if [[ -f "${m_log}" ]]; then
        parse_cleanup_from_log "${m_log}"
    else
        parse_cleanup_from_combined_log "${combined_log}" 0
    fi
    m_status="${CLEANUP_STATUS}"; m_detail="${CLEANUP_DETAIL}"
    if [[ -f "${w_log}" ]]; then
        parse_cleanup_from_log "${w_log}"
    else
        parse_cleanup_from_combined_log "${combined_log}" 1
    fi
    w_status="${CLEANUP_STATUS}"; w_detail="${CLEANUP_DETAIL}"
    if [[ "${m_status}" == "PASS" && "${w_status}" == "PASS" ]]; then
        CLEANUP_STATUS="PASS"
        CLEANUP_DETAIL="m:${m_detail} w:${w_detail}"
    elif [[ "${m_status}" == "UNKNOWN" || "${w_status}" == "UNKNOWN" ]]; then
        CLEANUP_STATUS="UNKNOWN"
        CLEANUP_DETAIL="m:${m_detail} w:${w_detail}"
    else
        CLEANUP_STATUS="FAIL"
        local parts=()
        [[ "${m_status}" != "PASS" ]] && parts+=("master:${m_detail}")
        [[ "${w_status}" != "PASS" ]] && parts+=("worker:${w_detail}")
        CLEANUP_DETAIL="${parts[*]}"
    fi
}

# ---------------------------------------------------------------------------
# analyse_run: read the combined log + evidence files, build a verdict row.
# Sets globals ROW_FAULT_RC, ROW_HIT, ROW_TARGET, ROW_SURVIVORS, ROW_MASK,
# ROW_LATENCY, ROW_TB, ROW_CLEANUP, ROW_VERDICT. Same checks as the 1-node
# sweep; the only difference is we operate on the COMBINED log which mixes
# master+worker stdout (both ranks' [rank R] elastic.py output, kill line,
# MASK DETECTED, WORKER DONE).
# ---------------------------------------------------------------------------
analyse_run() {
    local timing="$1"
    local log="$2"
    local fault_rc="$3"
    local expected_survivors="$4"
    local cleanup_status="$5"
    local cleanup_detail="$6"

    local hit="N/A"
    local target_match="N/A"
    local exp_target="${EXPECTED_TARGET[${timing}]:-}"
    if [[ -n "${exp_target}" ]]; then
        local evidence_glob="${EVIDENCE_DIR_HOST}/in_kernel_fault_rank*_${timing}_pid*.log"
        # shellcheck disable=SC2206
        local evidence_files=( ${evidence_glob} )
        if [[ -e "${evidence_files[0]}" ]]; then
            if grep -q HIT_IN_KERNEL_WINDOW "${evidence_files[@]}"; then
                hit="HIT"
            elif grep -q MISSED_IN_KERNEL_TIMING "${evidence_files[@]}"; then
                hit="MISSED"
            elif grep -q IN_KERNEL_MARKER_TIMEOUT "${evidence_files[@]}"; then
                hit="TIMEOUT"
            else
                hit="NO_VERDICT"
            fi
            local got_target
            got_target=$(sed -n 's/^target=//p' "${evidence_files[@]}" | head -n 1)
            if [[ "${got_target}" == "${exp_target}" ]]; then
                target_match="ok(${got_target})"
            else
                target_match="MISMATCH(want=${exp_target},got=${got_target:-none})"
            fi
        else
            hit="NO_EVIDENCE"
        fi
    fi

    local done_count
    done_count=$(grep -c '\[rank [0-9]\+\] WORKER DONE survived=true' "${log}" || true)
    local survivors_field="${done_count}/${expected_survivors}"

    local tb_count
    tb_count=$(grep -E -c 'Traceback|AssertionError|RuntimeError' "${log}" || true)

    local mask_pass=0 mask_total=0
    while IFS= read -r line; do
        local p c
        p=$(printf '%s' "${line}" | sed -n 's/.*passes=\([0-9]\+\).*/\1/p')
        c=$(printf '%s' "${line}" | sed -n 's/.*calls=\([0-9]\+\).*/\1/p')
        if [[ -n "${p}" && -n "${c}" ]]; then
            mask_pass=$((mask_pass + p))
            mask_total=$((mask_total + c))
        fi
    done < <(grep '\[rank [0-9]\+\] MASK CHECK SUMMARY' "${log}" || true)
    local mask_field
    if [[ "${mask_total}" -eq 0 ]]; then
        mask_field="n/a"
    elif [[ "${mask_pass}" -eq "${mask_total}" ]]; then
        mask_field="${mask_pass}/${mask_total}"
    else
        mask_field="${mask_pass}/${mask_total}!"
    fi

    local detect_ms="n/a"
    local kill_ns=""
    if [[ -n "${exp_target}" ]]; then
        local evidence_glob_for_ts="${EVIDENCE_DIR_HOST}/in_kernel_fault_rank*_${timing}_pid*.log"
        # shellcheck disable=SC2206
        local evidence_for_ts=( ${evidence_glob_for_ts} )
        if [[ -e "${evidence_for_ts[0]}" ]]; then
            kill_ns=$(grep -h '^timestamp_ns=' "${evidence_for_ts[@]}" 2>/dev/null \
                | sed 's/^timestamp_ns=//' | head -n 1)
        fi
    else
        kill_ns=$(grep -E '\[rank [0-9]+\] Killing rank at ' "${log}" \
            | sed -n 's/.*timestamp_ns=\([0-9]\+\).*/\1/p' | head -n 1)
    fi
    if [[ -n "${kill_ns}" ]]; then
        local det_ns
        det_ns=$(grep '\[rank [0-9]\+\] MASK DETECTED ' "${log}" 2>/dev/null \
            | sed -n 's/.*timestamp_ns=\([0-9]\+\).*/\1/p' \
            | awk -v k="${kill_ns}" '$1+0 >= k+0' \
            | sort -n | head -n 1)
        if [[ -n "${det_ns}" ]]; then
            detect_ms=$(awk -v a="${kill_ns}" -v b="${det_ns}" 'BEGIN{printf "%.1f", (b-a)/1e6}')
        fi
    fi

    local verdict="PASS"
    local fail_reasons=()
    [[ "${fault_rc}" -eq 0 ]] || fail_reasons+=("fault_rc=${fault_rc}")
    [[ "${done_count}" -eq "${expected_survivors}" ]] || fail_reasons+=("survivors=${done_count}/${expected_survivors}")
    [[ "${tb_count}" -eq 0 ]] || fail_reasons+=("tracebacks=${tb_count}")
    if [[ "${mask_total}" -gt 0 && "${mask_pass}" -ne "${mask_total}" ]]; then
        fail_reasons+=("mask=${mask_pass}/${mask_total}")
    fi
    if [[ -n "${exp_target}" ]]; then
        [[ "${hit}" == "HIT" ]] || fail_reasons+=("hit=${hit}")
        [[ "${target_match}" == ok* ]] || fail_reasons+=("target=${target_match}")
    fi
    [[ "${cleanup_status}" == "PASS" ]] || fail_reasons+=("cleanup=${cleanup_detail}")
    if [[ "${#fail_reasons[@]}" -gt 0 ]]; then
        verdict="FAIL(${fail_reasons[*]})"
    fi

    ROW_FAULT_RC="${fault_rc}"
    ROW_HIT="${hit}"
    ROW_TARGET="${target_match}"
    ROW_SURVIVORS="${survivors_field}"
    ROW_MASK="${mask_field}"
    ROW_LATENCY="${detect_ms}"
    ROW_TB="${tb_count}"
    ROW_CLEANUP="${cleanup_detail}"
    ROW_VERDICT="${verdict}"
}

# ---------------------------------------------------------------------------
# SUMMARY header
# ---------------------------------------------------------------------------
{
    echo "# Two-node NVLink fault-tolerance sweep"
    echo
    echo "- Started UTC: \`${UTC}\`"
    echo "- Job: \`${JOB_TAG}\`"
    echo "- Nodes: \`master=${MASTER_HOST}\`, \`worker=${WORKER_HOST}\`"
    echo "- Victim plan: \`${VICTIM_PLAN}\`"
    echo "- Baseline plan: \`${BASELINE_PLAN}\`"
    echo "- Mem leak tolerance (per node): \`${MEM_LEAK_MIB}\` MiB"
    echo "- Settle wait: \`${SETTLE_SECONDS}\` s"
    echo "- Iterations per timing: \`${ITERATIONS}\`"
    echo "- In-kernel spin cycles: \`${SPIN_CYCLES}\`"
    echo "- Run dir: \`${RUN_DIR_HOST}\`"
    echo
} > "${SUMMARY}"

echo "[2node-sweep] verifying build..." >&2
if ! verify_build; then
    log_summary "Build sanity FAILED. See \`build_probe.log\`."
    exit 1
fi
log_summary "- Build sanity: PASS"

EXPECTED_SURVIVORS_FAULT=$(expected_survivors_for_plan "${VICTIM_PLAN}")
EXPECTED_SURVIVORS_BASE=$(expected_survivors_for_plan "${BASELINE_PLAN}")
log_summary "- Expected survivors (fault / baseline): \`${EXPECTED_SURVIVORS_FAULT}\` / \`${EXPECTED_SURVIVORS_BASE}\`"

# ---------------------------------------------------------------------------
# Initial baseline (no fault)
# ---------------------------------------------------------------------------
echo "[2node-sweep] initial baseline..." >&2
init_log="${RUN_DIR_HOST}/initial_baseline.log"
run_one "${BASELINE_PLAN}" "${init_log}" "initial_baseline" 1
init_rc=$?
init_done=$(grep -c '\[rank [0-9]\+\] WORKER DONE survived=true' "${init_log}" || true)
if [[ "${init_rc}" -ne 0 || "${init_done}" -ne "${EXPECTED_SURVIVORS_BASE}" ]]; then
    log_summary "- Initial baseline: FAIL (rc=${init_rc}, survivors=${init_done}/${EXPECTED_SURVIVORS_BASE}). See \`initial_baseline.log\`."
    exit 1
fi
log_summary "- Initial baseline: PASS (rc=0, survivors=${init_done}/${EXPECTED_SURVIVORS_BASE})"

log_summary ""
log_summary "## Per-timing results"

OVERALL=0

for timing in "${TIMINGS_ARR[@]}"; do
    for iter in $(seq 1 "${ITERATIONS}"); do
        echo "[2node-sweep] ${timing} iter=${iter}/${ITERATIONS}" >&2
        # Clear prior evidence for this timing so we read fresh state.
        rm -f "${EVIDENCE_DIR_HOST}"/in_kernel_fault_rank*_"${timing}"_pid*.log

        log="${RUN_DIR_HOST}/${timing}__iter${iter}.log"
        is_in_kernel=0
        if [[ -n "${EXPECTED_TARGET[${timing}]:-}" ]]; then
            is_in_kernel=1
            run_one "${VICTIM_PLAN}" "${log}" "${timing}" "${iter}" \
                --fault-kill-timing "${timing}" \
                --in-kernel-fault-spin-cycles "${SPIN_CYCLES}"
        else
            run_one "${VICTIM_PLAN}" "${log}" "${timing}" "${iter}" \
                --fault-kill-timing "${timing}"
        fi
        fault_rc=$?

        # Cleanup verdict comes from the per-rank logs written by step.sh.
        parse_cleanup_combined "${timing}" "${iter}"

        analyse_run "${timing}" "${log}" "${fault_rc}" \
            "${EXPECTED_SURVIVORS_FAULT}" "${CLEANUP_STATUS}" "${CLEANUP_DETAIL}"

        post_log="${RUN_DIR_HOST}/${timing}__iter${iter}__post_baseline.log"
        run_one "${BASELINE_PLAN}" "${post_log}" "${timing}_post_baseline" "${iter}"
        post_rc=$?
        post_done=$(grep -c '\[rank [0-9]\+\] WORKER DONE survived=true' "${post_log}" || true)
        post_field="rc=${post_rc} survivors=${post_done}/${EXPECTED_SURVIVORS_BASE}"
        if [[ "${post_rc}" -ne 0 || "${post_done}" -ne "${EXPECTED_SURVIVORS_BASE}" ]]; then
            ROW_VERDICT="FAIL(post_baseline=${post_field})"
        fi

        log_summary ""
        log_summary "### \`${timing}\` (run ${iter})"
        log_summary ""
        log_summary "- Fault run exit code: \`${ROW_FAULT_RC}\`"
        if (( is_in_kernel == 1 )); then
            log_summary "- In-kernel spin cycles: \`${SPIN_CYCLES}\`"
            log_summary "- Marker HIT verdict: ${ROW_HIT}"
            log_summary "- In-kernel target slot: ${ROW_TARGET}"
        fi
        log_summary "- Survivors (observed/expected): ${ROW_SURVIVORS}"
        log_summary "- Mask-check passes: ${ROW_MASK}"
        log_summary "- Mask propagation latency: ${ROW_LATENCY} ms"
        log_summary "- Survivor tracebacks: ${ROW_TB}"
        log_summary "- Cleanup (master + worker, per-node): ${ROW_CLEANUP}"
        log_summary "- Post-fault baseline: ${post_field}"
        log_summary "- Verdict: ${ROW_VERDICT}"

        [[ "${ROW_VERDICT}" != PASS* ]] && OVERALL=1
    done
done

log_summary ""
log_summary "## Final"
log_summary ""
if [[ "${OVERALL}" -eq 0 ]]; then
    log_summary "- Overall: **PASS**"
else
    log_summary "- Overall: **FAIL** (grep for FAIL above)"
fi
log_summary "- Finished UTC: \`$(date -u +%Y%m%d_%H%M%S)\`"

echo "[2node-sweep] done; see ${SUMMARY}" >&2
exit "${OVERALL}"
