#!/usr/bin/env bash
# Single-node 4-GPU run of test_ht.py against an EP install at /workspace/nixl/install.
#
# Designed for the personal workflow on davidyogev's `ep-ht-nvl-island-size-4` branch:
# launch a Pyxis container from lishapira's latest.sqsh with the patched source
# bind-mounted at /workspace/nixl, build via meson+ninja directly (NOT .gitlab/build.sh,
# which deletes its TMPDIR before invoking nvcc), then invoke this script.
#
# Example end-to-end:
#   srun -A <acct> --partition=<part> --nodes=1 --gres=gpu:4 \
#        --container-image=/lustre/.../dyogev/latest.sqsh \
#        --container-mounts=/lustre/.../dyogev/nixl:/workspace/nixl \
#        --pty bash
#   # inside the container, after meson+ninja install:
#   bash examples/device/ep/tests/run_test_ht_single_node.sh
#
# Why the env tweaks below:
#   - UCX_TLS=^cuda_ipc (set by .gitlab/build.sh for multi-node CI) leaves the
#     device-side UCX path with no in-node GPU transport. Must be unset for a
#     single-island run, otherwise prepMemView fails with "Invalid parameter".
#   - PYTHONPATH puts our install ahead of /workspace/.venv so `import nixl_cu13`
#     loads the patched bindings, not the prebuilt ones from latest.sqsh.
#   - LD_LIBRARY_PATH does the same for libnixl.so.
#   - NIXL_PLUGIN_DIR makes sure plugins are dlopen'd from our install.

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

unset UCX_TLS
export PYTHONPATH=/workspace/nixl/install/lib/python3/dist-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/workspace/nixl/install/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export NIXL_PLUGIN_DIR=/workspace/nixl/install/lib/x86_64-linux-gnu/plugins

cd /workspace/nixl
exec python3 examples/device/ep/tests/test_ht.py --num-processes 4 "$@"
