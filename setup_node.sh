#!/usr/bin/env bash
# nixl-4165b16/setup_node.sh -- per-clone version of the GB300/GB200 container
# bring-up script. Sources `build_nixl_aarch64.sh env` so PYTHONPATH /
# LD_LIBRARY_PATH / NIXL_PLUGIN_DIR resolve `nixl_ep` from THIS clone's
# build+install (so the in-kernel-marker C++ binding compiled at this commit
# is actually loaded), while reusing Lior's prebuilt UCX at
# /workspace/lishapira/ucx/install.
#
# Source (do NOT execute):
#   source /workspace/dyogev/nixl-4165b16/setup_node.sh
#
# Mirrors /workspace/dyogev/setup_node.sh exactly, only with NIXL_SRC pointed
# at the 4165b16 clone instead of dyogev/nixl.

set +e
unset NIXL_SRC NIXL_BUILD NIXL_PREFIX PYTHONPATH UCX_HOME UCX_PREFIX

if [[ -n "${LD_LIBRARY_PATH-}" ]]; then
    new_ld=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cuda/compat' | grep -v '^$' | paste -sd:)
    export LD_LIBRARY_PATH="$new_ld"
fi
export UCX_WARN_UNUSED_ENV_VARS=n

export NIXL_SRC=/workspace/dyogev/nixl-4165b16
export NIXL_BUILD=$NIXL_SRC/build
export NIXL_PREFIX=$NIXL_SRC/install
export UCX_PREFIX=/workspace/lishapira/ucx/install

source /workspace/lishapira/build_nixl_aarch64.sh env
set +euo pipefail
