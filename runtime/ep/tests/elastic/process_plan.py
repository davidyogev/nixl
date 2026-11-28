# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch and reap the GPU workers of an elastic test plan, one supervisor per node."""

import json
import multiprocessing
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def run_process_plan(args, plan, world_ranks, max_so, max_su, worker, write_results):
    node = int(os.getenv("RANK", "0"))
    nodes = int(os.getenv("WORLD_SIZE", "1"))
    assert 0 <= node < nodes and len(world_ranks) == nodes * args.num_processes
    participating = set().union(*plan)
    owners = {
        index // args.num_processes
        for index, rank in enumerate(world_ranks)
        if rank in participating
    }
    if node and node not in owners:
        return

    # The CPU supervisor owns the server even when GPU rank zero is absent.
    store = dist.TCPStore(
        os.getenv("SERVER_ADDR", "127.0.0.1"),
        int(os.getenv("SERVER_PORT", "8361")),
        world_size=None,
        is_master=node == 0,
        timeout=timedelta(seconds=args.store_timeout_secs),
        wait_for_workers=False,
    )
    spawn = multiprocessing.get_context("spawn")
    local = world_ranks[node * args.num_processes : (node + 1) * args.num_processes]
    children = {}

    def wait_for(keys):
        # Poll the children as well as the Store so a failed worker wakes the
        # other supervisors even when its peers are blocked inside a collective.
        deadline = time.monotonic() + args.store_timeout_secs
        while True:
            if store.check(["failure"]):
                raise RuntimeError(store.get("failure").decode())
            for rank, (process, last_phase) in list(children.items()):
                if process.exitcode is None:
                    continue
                process.join()
                code, pid = process.exitcode, process.pid
                process.close()
                del children[rank]
                if code != 0 or not store.check(
                    [f"phase/{last_phase}/complete/{rank}"]
                ):
                    raise RuntimeError(
                        f"Worker rank {rank} on node {node} exited with code {code}; "
                        f"expected successful completion through phase {last_phase}"
                    )
                print(
                    f"Phase {last_phase} finished: rank {rank} exited (PID {pid})",
                    flush=True,
                )
                store.set(f"phase/{last_phase}/exit/{rank}", "1")
            if not keys or store.check(keys):
                return
            if time.monotonic() >= deadline:
                missing = [key for key in keys if not store.check([key])]
                raise TimeoutError(f"Node {node} timed out waiting for {missing}")
            time.sleep(0.05)

    try:
        for i, phase in enumerate(plan):
            if node == 0:
                store.set(f"phase/{i}/start", "1")
            wait_for([f"phase/{i}/start"])
            previous = plan[i - 1] if i else ()
            for local_rank, rank in enumerate(local):
                if rank not in phase or rank in previous:
                    continue
                # A worker serves one contiguous run of phases on the same GPU.
                last_phase = i
                while last_phase + 1 < len(plan) and rank in plan[last_phase + 1]:
                    last_phase += 1
                process = spawn.Process(
                    target=worker,
                    args=(local_rank, rank, i, plan, max_so, max_su, args),
                )
                process.start()
                children[rank] = (process, last_phase)
                print(f"Phase {i}: started rank {rank} (PID {process.pid})", flush=True)

            if node == 0:
                following = plan[i + 1] if i + 1 < len(plan) else ()
                complete = [f"phase/{i}/complete/{rank}" for rank in phase]
                exited = [
                    f"phase/{i}/exit/{rank}" for rank in phase if rank not in following
                ]
                wait_for(complete + exited)

        if node == 0:
            if args.save_output:
                dimensions = (
                    [max_so, max_su] if max_so else json.loads(store.get("geometry"))
                )
                ranks = [
                    dict(
                        world_rank=rank,
                        phases=[
                            json.loads(store.get(f"phase/{i}/complete/{rank}"))
                            for i, phase in enumerate(plan)
                            if rank in phase
                        ],
                    )
                    for rank in world_ranks
                ]
                write_results(
                    args.save_output,
                    dict(
                        vars(args),
                        maximum_world=dimensions,
                        torch_version=str(torch.__version__),
                    ),
                    plan,
                    ranks,
                )
            store.set("finished", "1")
            wait_for([f"finished/{owner}" for owner in sorted(owners) if owner != 0])
            print("Adaptive elastic test passed", flush=True)
        else:
            wait_for(["finished"])
            store.set(f"finished/{node}", "1")
    except BaseException as exc:
        # Preserve the first failure, including a local worker's exit status.
        try:
            store.compare_set("failure", "", f"Node {node}: {exc}")
        except RuntimeError:
            pass  # The Store owner may have already reported the failure and exited.
        raise
    finally:
        for process, _ in children.values():
            if process.is_alive():
                process.terminate()
        for process, _ in children.values():
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
