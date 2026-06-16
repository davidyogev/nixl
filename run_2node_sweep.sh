#!/usr/bin/env bash
# Driver for the 2-node NVLink fault-tolerance sweep on this dyogev clone of
# nixl @ 4165b16. Stays on the login node and lets the canonical 2-node
# sweep orchestrator launch each per-timing srun step inside the container
# image on the allocated compute nodes.
#
# Phase 1 (build): IF this clone's install lacks the in-kernel marker
# symbols (the default; lishapira's lustre install is stale), do a single
# `srun --nodes=1` step inside the container on the master node that runs
# build_nixl_aarch64.sh against THIS source. The sweep's verify_build will
# fail-fast if this step gets skipped for the wrong reason.
#
# Phase 2 (sweep): hand off to run_nvlink_fault_tolerance_2node_sweep.sh
# with FT_SETUP_NODE_SH / FT_TEST_DIR pointed at this clone (so step.sh
# loads OUR install, not lishapira's).
#
# Usage:
#   1. Acquire a 2-node salloc on the gb200 partition (--exclusive).
#      The two nodes MUST share an inter-node NVLink (MNNVL) fabric for
#      these tests to be meaningful -- without it, NCCL/UCX would silently
#      fall back to IB. We DON'T require a SLURM constraint up front: the
#      driver runs a Phase 0 nvidia-smi MNNVL probe and fail-fasts if the
#      allocation lands on two different NVL fabrics. Just re-salloc and
#      try again (usually the scheduler picks better the second time).
#      Optionally, pin the allocation up front to skip the retry roulette:
#          --constraint=nvlblk<NN>           (a single block, slowest queue)
#          --constraint='nvlblk07|nvlblk08'  (any of N blocks, faster queue)
#      Find candidate blocks (sorted by current idle count):
#          sinfo -p gb200 -h -o "%N %t %f" | awk '$2=="idle"' \
#            | grep -oE 'nvlblk[0-9]+' | sort | uniq -c | sort -rn | head
#      Example salloc:
#          salloc -N2 -p gb200 -A network_research_advdev -t 00:30:00 \
#            --exclusive -J network_research_advdev-nixl_ft.<tag> \
#            bash -c 'bash /lustre/fsw/network_research_advdev/dyogev/nixl-4165b16/run_2node_sweep.sh'
#   2. From inside that allocation (so SLURM_JOB_ID/SLURM_JOB_NODELIST are
#      populated), bash this script.
#
# Env overrides:
#   SKIP_BUILD=1          do not rebuild even if marker symbols are missing.
#   FORCE_BUILD=1         rebuild even if symbols are present.
#   NIXL_REF              git ref to build (default: detached HEAD of the
#                         clone, which is 4165b16 for the initial checkout).
#   TIMINGS               forwarded to the sweep (subset for smoke test).
#   SKIP_NVLINK_CHECK=1   bypass the Phase 0 MNNVL probe (debug only).

set -uo pipefail

: "${SLURM_JOB_ID:?need an active 2-node SLURM allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST must be set by salloc}"

DYOGEV_DIR=/lustre/fsw/network_research_advdev/dyogev
LISHAPIRA_DIR=/lustre/fsw/network_research_advdev/lishapira
SQSH="${DYOGEV_DIR}/nixl-hybrid-ep-cuda2.sqsh"
CLONE_HOST="${DYOGEV_DIR}/nixl-4165b16"
CLONE_CONTAINER="/workspace/dyogev/nixl-4165b16"
TEST_DIR="${CLONE_HOST}/examples/device/ep/tests/elastic"
INSTALL_HOST="${CLONE_HOST}/install"

[[ -f "${SQSH}" ]] || { echo "missing container image: ${SQSH}" >&2; exit 2; }
[[ -d "${TEST_DIR}" ]] || { echo "missing test dir: ${TEST_DIR}" >&2; exit 2; }

UTC=$(date -u +%Y%m%d_%H%M%S)
NODES_TAG=$(printf '%s' "${SLURM_JOB_NODELIST}" | tr -d '[]' | tr ',-' '_')
RUN_DIR="${TEST_DIR}/results/2node_${UTC}_job${SLURM_JOB_ID}_${NODES_TAG}"
mkdir -p "${RUN_DIR}"
# Open RUN_DIR perms so step.sh inside the container (running as the launching
# user, who is not necessarily in this tree's primary group) can create
# per-rank tee logs. mkdir defaults to 0755+setgid, which leaves "others" with
# no write even when the parent is 2777.
chmod 2777 "${RUN_DIR}" 2>/dev/null || true
DRIVER_LOG="${RUN_DIR}.console.log"
BUILD_LOG="${RUN_DIR}/build.log"

# The 2-node orchestrator relies on a single host<->container prefix pair
# that maps the lustre RUN_DIR (and the step script) to a path visible
# inside the container. Default in the sweep is lishapira's prefix; override
# here so our results live under dyogev/.
export CONTAINER_IMAGE="${SQSH}"
export CONTAINER_MOUNTS="${LISHAPIRA_DIR}:/workspace/lishapira,${DYOGEV_DIR}:/workspace/dyogev"
export LUSTRE_HOST_PREFIX="${DYOGEV_DIR}"
export LUSTRE_CONTAINER_PREFIX="/workspace/dyogev"
export RUN_DIR
export FT_SETUP_NODE_SH="${CLONE_CONTAINER}/setup_node.sh"
export FT_TEST_DIR="${CLONE_CONTAINER}/examples/device/ep/tests/elastic"

# Resolve master node for the build srun.
MASTER_HOST=$(printf '%s' "${SLURM_JOB_NODELIST}" \
    | sed -nE 's/^([^[]+)\[([0-9]+)[-,].*$/\1\2/p; t end; s/^([^[]+\[)?([^][,]+)\]?$/\2/p; :end' \
    | head -n 1)
if [[ -z "${MASTER_HOST}" ]] && command -v scontrol >/dev/null 2>&1; then
    MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null | head -n 1)
fi
[[ -n "${MASTER_HOST}" ]] || { echo "[driver] FATAL: could not resolve master host from ${SLURM_JOB_NODELIST}" >&2; exit 2; }

echo "[driver] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[driver] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[driver] MASTER_HOST=${MASTER_HOST}"

# scontrol nvlblk label (FYI only -- the real gate is the Phase 0 MNNVL probe
# below; the SLURM feature label can be stale or missing on some clusters).
if command -v scontrol >/dev/null 2>&1; then
    NVL_BLOCKS=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null \
        | while read -r n; do
              scontrol show node "${n}" 2>/dev/null \
                | grep -oE 'nvlblk[0-9]+' | head -n 1
          done | sort -u | tr '\n' ' ')
    NVL_BLOCKS=${NVL_BLOCKS% }
    if [[ -n "${NVL_BLOCKS}" ]]; then
        if [[ "${NVL_BLOCKS}" == *" "* ]]; then
            echo "[driver] FYI: scontrol shows allocation spans NVL blocks: ${NVL_BLOCKS} (Phase 0 will check)"
        else
            echo "[driver] NVL_BLOCK_LABEL=${NVL_BLOCKS}"
        fi
    fi
fi
echo "[driver] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "[driver] CONTAINER_MOUNTS=${CONTAINER_MOUNTS}"
echo "[driver] CLONE=${CLONE_CONTAINER}"
echo "[driver] FT_SETUP_NODE_SH=${FT_SETUP_NODE_SH}"
echo "[driver] FT_TEST_DIR=${FT_TEST_DIR}"
echo "[driver] RUN_DIR=${RUN_DIR}"
echo "[driver] driver log: ${DRIVER_LOG}"
echo "[driver] build log:  ${BUILD_LOG}"

# ---------------------------------------------------------------------------
# Phase 0: verify inter-node NVLink (MNNVL) before doing anything expensive.
#
# These tests exercise multi-node NVLink (MNNVL). Without it, NCCL/UCX would
# silently fall back to IB and the in-kernel fault tests would produce
# meaningless results. The canonical hardware-truth check is to read every
# GPU's `ClusterUUID` and `CliqueId` -- two GPUs are MNNVL-coupled iff they
# share both fields.
#
# We run the probe inside the test container (the host nvidia-smi binary on
# Lyris compute nodes is older and rejects `-d FABRIC`; the container ships
# a current binary that talks to the host driver via mapped libnvidia-ml).
# We try `-q -d FABRIC` first and fall back to plain `-q` if the binary
# version mismatch ever bites us in the future. Output is filtered to just
# the relevant fields.
#
# Override with SKIP_NVLINK_CHECK=1 for debug.
# ---------------------------------------------------------------------------
NVLINK_PROBE_DIR="${RUN_DIR}/nvlink_probe"
mkdir -p "${NVLINK_PROBE_DIR}"
chmod 2777 "${NVLINK_PROBE_DIR}" 2>/dev/null || true

if [[ "${SKIP_NVLINK_CHECK:-0}" == "1" ]]; then
    echo "[driver] SKIP_NVLINK_CHECK=1 -> skipping MNNVL fabric/clique check"
else
    echo "[driver] === Phase 0: verify inter-node NVLink (MNNVL) ==="
    # Run inside the container -- the host nvidia-smi on Lyris compute nodes
    # is older than the driver and rejects `-d FABRIC`; the container ships
    # with a current nvidia-smi that supports it. NVLINK_PROBE_DIR is on
    # Lustre and visible at the same path inside the container.
    NVLINK_PROBE_DIR_INSIDE="${NVLINK_PROBE_DIR/${LUSTRE_HOST_PREFIX}/${LUSTRE_CONTAINER_PREFIX}}"
    nvl_probe_rc=0
    srun --jobid="${SLURM_JOB_ID}" --overlap \
        --nodes=2 --ntasks-per-node=1 \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        --container-workdir=/workspace/dyogev \
        --no-container-mount-home \
        --output="${NVLINK_PROBE_DIR}/srun.%n.%t.log" \
        --export=ALL,NVLINK_PROBE_DIR_INSIDE="${NVLINK_PROBE_DIR_INSIDE}" \
        bash -c '
            set -uo pipefail
            out="${NVLINK_PROBE_DIR_INSIDE}/nvlink_${SLURM_NODEID}_$(hostname).log"
            { echo "=== nvidia-smi -q -d FABRIC ==="; nvidia-smi -q -d FABRIC; } > "${out}" 2>&1
            rc=$?
            if (( rc != 0 )); then
                echo ""              >> "${out}"
                echo "(-d FABRIC rc=${rc}; falling back to plain -q)" >> "${out}"
                echo "=== nvidia-smi -q (filtered) ==="               >> "${out}"
                nvidia-smi -q 2>&1 | grep -E "Cluster ?UUID|Clique ?Id|Fabric" >> "${out}"
            fi
            exit 0
        ' \
        || nvl_probe_rc=$?

    if (( nvl_probe_rc != 0 )); then
        echo "[driver] FATAL: srun for NVLink probe failed (rc=${nvl_probe_rc})" >&2
        ls -la "${NVLINK_PROBE_DIR}" >&2 || true
        for f in "${NVLINK_PROBE_DIR}"/nvlink_*.log "${NVLINK_PROBE_DIR}"/srun.*.log; do
            [[ -s "${f}" ]] || continue
            echo "--- ${f} ---" >&2
            sed -n '1,30p' "${f}" >&2
        done
        exit 3
    fi

    declare -a UUIDS=() CLIQUES=() HOSTS=()
    for f in "${NVLINK_PROBE_DIR}"/nvlink_*.log; do
        [[ -f "${f}" ]] || continue
        host=$(basename "${f}" .log | sed -E 's/^nvlink_[0-9]+_//')
        uuid=$(grep -E "Cluster ?UUID" "${f}" | head -n1 | awk -F: '{print $NF}' | tr -d '[:space:]')
        clique=$(grep -E "Clique ?Id" "${f}" | head -n1 | awk -F: '{print $NF}' | tr -d '[:space:]')
        HOSTS+=("${host}")
        UUIDS+=("${uuid:-N/A}")
        CLIQUES+=("${clique:-N/A}")
        echo "[driver]   ${host}: ClusterUUID=${uuid:-N/A} CliqueId=${clique:-N/A}"
    done

    if (( ${#UUIDS[@]} == 0 )); then
        echo "[driver] FATAL: NVLink probe collected no per-node logs in ${NVLINK_PROBE_DIR}" >&2
        exit 5
    fi
    uniq_uuids=$(printf '%s\n' "${UUIDS[@]}" | sort -u | wc -l)
    uniq_cliques=$(printf '%s\n' "${CLIQUES[@]}" | sort -u | wc -l)

    if [[ "${UUIDS[0]:-N/A}" == "N/A" ]] || (( uniq_uuids != 1 )) || (( uniq_cliques != 1 )); then
        echo "[driver] FATAL: nodes are NOT in the same MNNVL fabric/clique." >&2
        echo "[driver]   ClusterUUID values seen: ${UUIDS[*]}" >&2
        echo "[driver]   CliqueId values seen:    ${CLIQUES[*]}" >&2
        echo "[driver]   Inter-node NVLink tests would fall back to IB and produce" >&2
        echo "[driver]   meaningless results. Re-salloc to retry, or pin with" >&2
        echo "[driver]   --constraint=nvlblk<NN> (or nvlblkA|nvlblkB|...), or set" >&2
        echo "[driver]   SKIP_NVLINK_CHECK=1 to force-run anyway." >&2
        exit 4
    fi
    echo "[driver] NVLink check PASS: both nodes share fabric (ClusterUUID=${UUIDS[0]} CliqueId=${CLIQUES[0]})"
fi

# ---------------------------------------------------------------------------
# Phase 1: build (probe + maybe rebuild)
# ---------------------------------------------------------------------------
need_build=0
if [[ "${FORCE_BUILD:-0}" == "1" ]]; then
    need_build=1
    echo "[driver] FORCE_BUILD=1 -> rebuilding"
elif [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "[driver] SKIP_BUILD=1 -> skipping rebuild check"
elif [[ ! -e "${INSTALL_HOST}/lib/python3/dist-packages/nixl_ep_cu13/nixl_ep_cpp.cpython-312-aarch64-linux-gnu.so" \
        && ! -e "${INSTALL_HOST}/lib/python3/dist-packages/nixl_ep/nixl_ep_cpp.cpython-312-aarch64-linux-gnu.so" ]]; then
    need_build=1
    echo "[driver] no built nixl_ep .so under ${INSTALL_HOST} -> rebuilding"
else
    # Probe symbols without an srun -- strings on lustre is fast enough.
    so_path=$(ls "${INSTALL_HOST}/lib/python3/dist-packages/nixl_ep_cu13/nixl_ep_cpp.cpython-"*.so 2>/dev/null \
              || ls "${INSTALL_HOST}/lib/python3/dist-packages/nixl_ep/nixl_ep_cpp.cpython-"*.so 2>/dev/null \
              | head -n 1)
    if ! strings "${so_path}" 2>/dev/null | grep -q enable_in_kernel_fault_marker; then
        need_build=1
        echo "[driver] ${so_path} lacks enable_in_kernel_fault_marker -> rebuilding"
    else
        echo "[driver] existing build at ${INSTALL_HOST} has marker symbols -> skipping rebuild"
    fi
fi

if (( need_build == 1 )); then
    echo "[driver] === Phase 1: build (1 srun on ${MASTER_HOST}, in container) ==="
    # Build script: clone-aware. NIXL_SRC pinned at our clone, NIXL_REF
    # forced to whatever the clone HEAD is so the build script doesn't
    # `git checkout main`. CLEAN=1 wipes any prior partial build.
    NIXL_REF_DEFAULT=$(git -C "${CLONE_HOST}" rev-parse HEAD 2>/dev/null || echo HEAD)
    NIXL_REF_TO_BUILD="${NIXL_REF:-${NIXL_REF_DEFAULT}}"
    BUILD_INSIDE='
set -uo pipefail
unset NIXL_PLUGIN_DIR PYTHONPATH UCX_HOME

export NIXL_SRC=/workspace/dyogev/nixl-4165b16
export NIXL_BUILD=$NIXL_SRC/build
export NIXL_PREFIX=$NIXL_SRC/install
export UCX_PREFIX=/workspace/lishapira/ucx/install
export NIXL_REPO=https://github.com/lishapira/nixl.git
export NIXL_REF='"${NIXL_REF_TO_BUILD}"'
export CLEAN=1
export JOBS=${JOBS:-$(nproc)}
export UCX_WARN_UNUSED_ENV_VARS=n

echo "[build] HEAD before: $(git -C "$NIXL_SRC" rev-parse --short HEAD 2>/dev/null || echo none)"
bash /workspace/lishapira/build_nixl_aarch64.sh
build_rc=$?
echo "[build] rc=${build_rc}"
exit ${build_rc}
'
    set -x
    srun --jobid="${SLURM_JOB_ID}" --overlap \
        --nodes=1 --ntasks=1 --nodelist="${MASTER_HOST}" \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        --container-workdir=/workspace/dyogev \
        --no-container-mount-home \
        bash -c "${BUILD_INSIDE}" \
        > "${BUILD_LOG}" 2>&1
    build_rc=$?
    set +x
    echo "[driver] build srun rc=${build_rc} (full log: ${BUILD_LOG})"
    if [[ "${build_rc}" -ne 0 ]]; then
        echo "[driver] build FAILED; aborting before sweep" >&2
        tail -n 60 "${BUILD_LOG}" >&2 || true
        exit "${build_rc}"
    fi
fi

# ---------------------------------------------------------------------------
# Phase 1.5: sanitize permissions before the sweep.
#
# A previous `chmod -R 2777 nixl-4165b16` set the setgid bit on every regular
# file (including .so files in install/). ld.so refuses to resolve symbols in
# setgid shared libraries when they live on a non-system path, which makes
# `import nixl_ep` fail silently in worker children -> elastic.py exits with
# "Worker processes failed: worker N (exit code 1)".
#
# Strip setgid from regular files only; leave the setgid bit on directories
# (we want it for group inheritance under our 2777 perms scheme).
# ---------------------------------------------------------------------------
strip_count=$(find "${CLONE_HOST}" -type f -perm -2000 -not -path '*/.git/*' 2>/dev/null | wc -l)
if [[ "${strip_count}" -gt 0 ]]; then
    echo "[driver] stripping setgid bit from ${strip_count} files in clone (saves dlopen)"
    find "${CLONE_HOST}" -type f -perm -2000 -not -path '*/.git/*' \
        -exec chmod g-s {} + 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Phase 2: sweep
# ---------------------------------------------------------------------------
echo "[driver] === Phase 2: 2-node sweep ==="
cd "${TEST_DIR}"
bash run_nvlink_fault_tolerance_2node_sweep.sh 2>&1 | tee "${DRIVER_LOG}"
rc=${PIPESTATUS[0]}
echo "[driver] sweep exited rc=${rc}"
echo "[driver] SUMMARY: ${RUN_DIR}/SUMMARY.md"
exit "${rc}"
