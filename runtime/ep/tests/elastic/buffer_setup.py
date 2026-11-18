# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit local construction, peer connection and activation for group-based tests."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from nixl.runtime.ep import ElasticBuffer
from nixl.runtime.ep.utils.nixl_bootstrap import NixlBootstrap, resolve_bootstrap_store


def get_logical_domain_size(
    group: dist.ProcessGroup,
    *,
    num_scaleup_ranks: Optional[int] = None,
    store: Optional[dist.Store] = None,
    store_timeout_secs: int = 300,
) -> Tuple[int, int]:
    """Discover uniform NVLink domains for group-based tests."""
    rank, num_ranks = group.rank(), group.size()
    bootstrap_store = resolve_bootstrap_store(
        group, store, timeout_seconds=store_timeout_secs
    )
    cohort = NixlBootstrap(
        bootstrap_store,
        rank,
        num_ranks,
        world_ranks=dist.get_process_group_ranks(group),
        num_scaleup_ranks=num_scaleup_ranks,
        timeout_seconds=store_timeout_secs,
    )
    return cohort.num_scaleout_ranks, cohort.num_scaleup_ranks


def create_group_buffer(group, *, buffer_type=ElasticBuffer, topology=None, **kwargs):
    store = resolve_bootstrap_store(group, kwargs.pop("store", None))
    scaleout, scaleup = topology or get_logical_domain_size(group, store=store)
    max_scaleout = kwargs.pop("max_num_scaleout_ranks", 0) or scaleout
    max_scaleup = kwargs.pop("max_num_scaleup_ranks", 0) or scaleup
    world = [
        (rank // scaleup) * max_scaleup + rank % scaleup for rank in range(group.size())
    ]
    buffer = buffer_type(
        world_rank=world[group.rank()],
        max_num_scaleout_ranks=max_scaleout,
        max_num_scaleup_ranks=max_scaleup,
        store=store,
        **kwargs,
    )
    buffer.connect_ranks([rank for rank in world if rank != buffer.world_rank])
    buffer.set_active_ranks(world)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    return buffer
