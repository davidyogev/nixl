# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch and reap the GPU workers of an elastic test plan."""

import json
import multiprocessing
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def run_process_plan(args, plan, world_ranks, max_so, max_su, worker, write_results):
    node = int(os.getenv("RANK", "0"))
    local = args.worker_ranks
    if local is None:
        local = world_ranks[node * args.num_processes : (node + 1) * args.num_processes]
    else:
        if not 0 <= args.join_phase < len(plan):
            raise ValueError("--join-phase is outside the plan")
        previous = plan[args.join_phase - 1] if args.join_phase else ()
        joining = set(plan[args.join_phase]).difference(previous)
        if not (
            0 < len(local) <= args.num_processes
            and len(set(local)) == len(local)
            and set(local) <= joining
        ):
            raise ValueError(
                "--worker-ranks must name distinct ranks joining at --join-phase, "
                "one per local GPU"
            )
    if node and not any(set(local).intersection(phase) for phase in plan):
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
    children = {}

    def wait_for(keys=()):
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
            if (keys and store.check(keys)) or (not keys and not children):
                return
            # A client waiting for its next launch or last child can span many
            # phases. The coordinator enforces a separate deadline per phase.
            if node == 0 and time.monotonic() >= deadline:
                missing = [key for key in keys if not store.check([key])]
                raise TimeoutError(f"Node {node} timed out waiting for {missing}")
            time.sleep(0.05)

    try:
        for i, phase in enumerate(plan):
            previous = plan[i - 1] if i else ()
            starts = [
                (ordinal, rank)
                for ordinal, rank in enumerate(local)
                if rank in phase
                and rank not in previous
                and (args.worker_ranks is None or i == args.join_phase)
            ]
            if node == 0:
                store.set(f"phase/{i}/start", "1")
            elif not starts:
                continue
            wait_for([f"phase/{i}/start"])
            for local_rank, rank in starts:
                if store.add(f"phase/{i}/owner/{rank}", 1) != 1:
                    raise ValueError(
                        f"Multiple launchers claimed rank {rank} in phase {i}"
                    )
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
            print("Adaptive elastic test passed", flush=True)
        else:
            # The final exit record is our last Store operation. The coordinator
            # can close the Store as soon as all GPU processes have been reaped.
            wait_for()
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
