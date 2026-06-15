# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import time

# Add elastic subdirectory to path for store_group import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "elastic"))
# noinspection PyUnresolvedReferences
import nixl_ep  # noqa: E402
import store_group  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from utils import (  # noqa: E402
    bench,
    bench_kineto,
    calc_diff,
    create_grouped_scores,
    init_dist,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
)

TCP_STORE_PORT = 9999


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: nixl_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups, num_topk, num_experts = (
        args.num_topk_groups,
        args.num_topk,
        args.num_experts,
    )

    # [dyogev patch] was hard-coded num_local_ranks == 4 (single-island branch);
    # 2x2 sanity run uses num_local_ranks == 2, 2 RDMA islands on one physical host.
    assert num_experts % num_ranks == 0 and num_local_ranks in (2, 4)
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}",
            flush=True,
        )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(
        group_scores, k=num_topk_groups, dim=-1, sorted=False
    ).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[
        1
    ]
    topk_idx = topk_idx.to(nixl_ep.topk_idx_t)
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    )
    topk_weights_pure_rand = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    )
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="cuda"
        )
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank_semantic = token_idx_in_rank >= 0
    # [dyogev patch] Pad the matrix to the kernel's padded row layout:
    # `[num_tokens, num_rdma_ranks * IS_TOKEN_IN_RANK_ISLAND_STRIDE]` (= 8 bytes
    # per island when num_local_ranks < 8) so the HT kernel can issue naturally-
    # aligned uint64 loads. Bytes beyond num_local_ranks within each island slice
    # are zero. Must mirror configs.cuh:IS_TOKEN_IN_RANK_ISLAND_STRIDE.
    IS_TOKEN_IN_RANK_ISLAND_STRIDE = 8 if num_local_ranks < 8 else num_local_ranks
    is_token_in_rank = torch.zeros(
        num_tokens, num_nodes, IS_TOKEN_IN_RANK_ISLAND_STRIDE,
        dtype=torch.bool, device="cuda",
    )
    is_token_in_rank[..., :num_local_ranks] = is_token_in_rank_semantic.view(
        num_tokens, num_nodes, num_local_ranks
    )
    is_token_in_rank = is_token_in_rank.view(
        num_tokens, num_nodes * IS_TOKEN_IN_RANK_ISLAND_STRIDE
    ).contiguous()
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    (
        ref_num_tokens_per_rank,
        ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    # [dyogev patch] runtime returns None for num_tokens_per_rdma_rank in intranode
    # settings (single NVL island, num_nodes==1); skip the comparison there.
    if ref_num_tokens_per_rdma_rank is not None:
        assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
        print("", flush=True)
    group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = nixl_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end="",
                        )
                    dispatch_args = {
                        "x": current_x,
                        "num_tokens_per_rank": num_tokens_per_rank,
                        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                        "is_token_in_rank": is_token_in_rank,
                        "num_tokens_per_expert": num_tokens_per_expert,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        dispatch_args.update(
                            {
                                "topk_idx": topk_idx,
                                "topk_weights": (
                                    topk_weights_pure_rand
                                    if current_x is x_pure_rand
                                    else topk_weights
                                ),
                            }
                        )
                    if previous_mode:
                        dispatch_args.update({"previous_event": buffer.capture()})
                    (
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        recv_num_tokens_per_expert_list,
                        handle,
                        event,
                    ) = buffer.ht_dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = (
                        per_token_cast_back(*recv_x)
                        if isinstance(recv_x, tuple)
                        else recv_x
                    )

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]
                    _expected_recv = gbl_num_tokens_per_rank[rank].item()
                    _actual_recv = recv_x.size(0)
                    _per_rank_actual_recv = torch.tensor(
                        [_actual_recv], dtype=torch.long, device="cuda"
                    )
                    _per_rank_gather = [
                        torch.zeros(1, dtype=torch.long, device="cuda")
                        for _ in range(num_ranks)
                    ]
                    dist.all_gather(_per_rank_gather, _per_rank_actual_recv, group=group)
                    _all_actual = [t.item() for t in _per_rank_gather]
                    _all_expected = gbl_num_tokens_per_rank.tolist()
                    if rank == 0:
                        print(
                            f"[diag] per-rank token receive (rank: expected -> actual, diff)\n"
                            + "\n".join(
                                f"  rank {r}: {_all_expected[r]:>6} -> {_all_actual[r]:>6}  "
                                f"diff={_all_actual[r]-_all_expected[r]:+}  "
                                f"({(_all_actual[r]/_all_expected[r]*100):.1f}%)"
                                if _all_expected[r] else
                                f"  rank {r}: {_all_expected[r]:>6} -> {_all_actual[r]:>6}  diff={_all_actual[r]-_all_expected[r]:+}"
                                for r in range(num_ranks)
                            ),
                            flush=True,
                        )
                    dist.barrier(group=group)
                    assert _expected_recv == _actual_recv, (
                        f"rank {rank}: expected {_expected_recv}, got {_actual_recv}"
                    )
                    _expected_expert_list = gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                    _diff_idx = [
                        i for i in range(len(_expected_expert_list))
                        if _expected_expert_list[i] != recv_num_tokens_per_expert_list[i]
                    ]
                    # Always print per-rank, so we see clean ranks too. Barrier
                    # before+after so SIGTERM after a failing rank doesn't clip
                    # the prints from other ranks.
                    dist.barrier(group=group)
                    print(
                        f"[diag rank={rank}] expected_sum={sum(_expected_expert_list)} "
                        f"got_sum={sum(recv_num_tokens_per_expert_list)} "
                        f"num_mismatch={len(_diff_idx)} "
                        f"first_mismatch_idx={_diff_idx[:8]}\n"
                        f"  expected[first8]={_expected_expert_list[:8]}\n"
                        f"  got[first8]     ={recv_num_tokens_per_expert_list[:8]}\n"
                        f"  expected[last8] ={_expected_expert_list[-8:]}\n"
                        f"  got[last8]      ={recv_num_tokens_per_expert_list[-8:]}",
                        flush=True,
                    )
                    dist.barrier(group=group)
                    assert _expected_expert_list == recv_num_tokens_per_expert_list, (
                        f"rank {rank}: expert-count mismatch, see [diag rank=...] lines"
                    )
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert recv_topk_idx is not None
                        assert recv_topk_weights is not None
                        assert (
                            recv_topk_idx.eq(-1)
                            | (
                                (recv_topk_idx >= 0)
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = (
                                recv_topk_weights.amax(dim=1, keepdim=True).expand_as(
                                    recv_topk_weights
                                )[recv_topk_idx.eq(-1)]
                            )
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {
                            "x": current_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        recv_x_cached, _, _, _, _, event = buffer.ht_dispatch(
                            **dispatch_args
                        )
                        event.current_stream_wait() if async_mode else ()
                        recv_x_cached = (
                            per_token_cast_back(*recv_x_cached)
                            if isinstance(recv_x_cached, tuple)
                            else recv_x_cached
                        )

                        if current_x is not x_pure_rand:
                            check_data(recv_x_cached, recv_gbl_rank_prefix_sum)

                        # Use cached result for combine
                        recv_x = recv_x_cached

                    # Test combine
                    bias_0 = torch.ones(
                        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
                    )
                    bias_1 = torch.randn(
                        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
                    )
                    combine_args = {
                        "x": recv_x,
                        "bias": (bias_0, bias_1),
                        "handle": handle,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        combine_args.update({"topk_weights": recv_topk_weights})
                    if previous_mode:
                        combine_args.update({"previous_event": buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.ht_combine(
                        **combine_args
                    )
                    event.current_stream_wait() if async_mode else ()

                    check_x = (
                        combined_x.float() - bias_0.float() - bias_1.float()
                    ) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = (
                            combined_topk_weights
                            if (current_x is x_pure_rand)
                            else (
                                combined_topk_weights
                                / is_token_in_rank.sum(dim=1).unsqueeze(1)
                            )
                        )
                        ref_topk_weights = (
                            topk_weights_pure_rand
                            if current_x is x_pure_rand
                            else topk_weights
                        )
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    # Sync all ranks before printing passed
                    group.barrier()
                    if local_rank == 0:
                        print(" passed", flush=True)
                    group.barrier()
    if local_rank == 0:
        print("", flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (
            (dispatch_bf16_rdma_send_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_rdma_send_bytes
        )
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = nixl_ep.Config(
                    num_sms,
                    nvl_chunk_size,
                    nvl_buffer_size,
                    rdma_chunk_size,
                    rdma_buffer_size,
                )
                tune_args = {"x": current_x, "handle": handle, "config": config}
                t, notify_t = bench_kineto(
                    lambda: buffer.ht_dispatch(**tune_args), ("dispatch", "notify")
                )
                if t < best_time:
                    best_time, best_results = t, (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        notify_t,
                    )
                if local_rank == 0:
                    print(
                        f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                        flush=True,
                    )
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)',  # type: ignore[index]
                flush=True,
            )
            print("", flush=True)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device="cuda")  # type: ignore[index]
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results)
                for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(
                all_best_fp8_results_list, best_dispatch_results, group=group
            )
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = nixl_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2], rdma_buffer_size)  # type: ignore[index]

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.ht_dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = nixl_ep.Config(
                num_sms,
                nvl_chunk_size,
                nvl_buffer_size,
                rdma_chunk_size,
                rdma_buffer_size,
            )
            tune_args = {"x": recv_x, "handle": handle, "config": config}
            t, notify_t = bench_kineto(
                lambda: buffer.ht_combine(**tune_args), ("combine", "notify")
            )
            if local_rank == 0:
                print(
                    f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                    flush=True,
                )
                if t < best_time:
                    best_time, best_results = t, (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        notify_t,
                    )

    if local_rank == 0:
        print(f"[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)", flush=True)  # type: ignore[index]
        print("", flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # Pin each process to a distinct GPU so NCCL does not see duplicate devices.
    # Use local_rank so NCCL gets correct device_id; avoid CUDA_VISIBLE_DEVICES
    # so that UCX/DOCA can see all GPUs for GPU-initiated RDMA when needed.
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank % 8)

    # [dyogev patch] Per-island fake hostname so the LD_PRELOAD shim in
    # island_hostname.so reports a unique hostname per NVL island. UCX uses
    # hostname comparison to classify endpoints as intra-node vs inter-node:
    # within an island (same fake hostname) UCX still picks cuda_ipc for the
    # device lane (preserving the NVL fast path); across islands (different
    # fake hostnames) UCX is eligible to use rc_gda, enabling real
    # kernel-issued RDMA on a single physical host.
    # Must be set before nixl_ep.Buffer() because UCX reads hostname during
    # UCP context construction. No-op if the LD_PRELOAD shim isn't loaded.
    _island_id = local_rank // num_local_ranks
    os.environ["UCX_FAKE_HOSTNAME"] = f"ep-island-{_island_id}"

    num_nodes = int(os.getenv("WORLD_SIZE", 1))

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    print(
        f"pid: {os.getpid()}, rank: {rank}, num_ranks: {num_ranks} ,local_rank: {local_rank}",
        flush=True,
    )
    if args.test_ll_compatibility:
        ll_num_experts = 256

    num_sms = 24
    num_qps_per_rank = max(
        num_sms // 2, ll_num_experts // num_ranks if args.test_ll_compatibility else 0
    )

    # Create TCPStore client for NIXL metadata exchange
    tcp_server = args.tcp_server if args.tcp_server else "127.0.0.1"
    tcp_store = store_group.create_client_store(
        master_addr=tcp_server,
        port=TCP_STORE_PORT,
    )

    # Initialize NIXL buffer with group (for IPC handles) and TCPStore (for NIXL metadata)
    print(
        f"pid: {os.getpid()}, rank: {rank}, num_ranks: {num_ranks}, initializing buffer",
        flush=True,
    )
    buffer = nixl_ep.Buffer(
        rank=rank,
        low_latency_mode=False,
        explicitly_destroy=True,
        group=group,
        tcp_store_group=tcp_store,
    )
    buffer.update_memory_buffers(
        num_ranks=num_ranks,
        num_experts_per_rank=num_qps_per_rank,
        num_nvl_bytes=int(2e9),
        num_rdma_bytes=int(1e9),
    )
    buffer.connect_ranks([i for i in range(num_ranks) if i != rank])

    # [dyogev patch] accept both the 4-GPU single-island (4,4) and the 2x2 single-host (2,4) layouts.
    assert num_local_ranks in (2, 4) and num_ranks == 4 and num_ranks % num_local_ranks == 0
    torch.manual_seed(rank + int(os.getenv("EP_SEED_OFFSET", "0")))

    for i in (num_sms,):
        test_main(
            args,
            i,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            buffer,
            group,
        )
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def run_server():
    _store = store_group.create_master_store(port=TCP_STORE_PORT)  # noqa: F841
    # Keep the server process alive while TCPStore serves requests
    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test high-throughput EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    # [dyogev patch] decoupled from --num-processes so a single host can pretend to be
    # multiple logical nodes (e.g. --num-processes 4 --num-local-ranks 2 + WORLD_SIZE=2
    # gives a 2x2 layout where the kernel routes 0<->1 and 2<->3 over NVLink and the
    # 0<->2/0<->3/1<->2/1<->3 pairs through the RDMA codepath).
    parser.add_argument(
        "--num-local-ranks",
        type=int,
        default=None,
        help="NIXL num_local_ranks (default: same as --num-processes)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk-groups",
        type=int,
        default=None,
        help="Number of top-k groups (default: `min(num_nodes, 4)`)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--test-ll-compatibility",
        action="store_true",
        help="whether to test compatibility with low-latency kernels",
    )
    parser.add_argument(
        "--tcp-server",
        type=str,
        help="TCP server address (for both TCPStore and rank server). If not set, both will be started locally.",
    )
    args = parser.parse_args()

    if not args.tcp_server:
        print("Starting TCPStore and rank server locally", flush=True)
        server_process = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_process.start()
        time.sleep(0.5)

    # Set default `num_topk_groups` if not provided
    if args.num_topk_groups is None:
        num_nodes = int(os.getenv("WORLD_SIZE", 1))
        args.num_topk_groups = min(num_nodes, 4)

    num_processes = args.num_processes
    # [dyogev patch] num_local_ranks defaults to num_processes (preserves the legacy invariant
    # nprocs == num_local_ranks); the 2x2 single-host run overrides it.
    num_local_ranks = args.num_local_ranks if args.num_local_ranks is not None else num_processes
    # 2-node run (WORLD_SIZE=2): run on both nodes with same MASTER_ADDR/MASTER_PORT; node1 needs --tcp-server <node0_ip>.
    # NVL/RDMA timeouts across nodes usually mean RDMA/IB/UCX between nodes is broken or slow (e.g. "accelerated IB support was not found" on one node).
    torch.multiprocessing.spawn(
        test_loop, args=(num_local_ranks, args), nprocs=num_processes
    )
