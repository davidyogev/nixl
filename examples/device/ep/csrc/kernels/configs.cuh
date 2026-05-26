/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// [dyogev patch] reduced from 8 for 2x2 single-host testing (2 NVL peers per
// island, 2 RDMA islands, 4 GPUs total). Pair within an island shares NVLink;
// pairs across islands traverse the kernel's RDMA codepath through nixlPut.
#define NUM_MAX_NVL_PEERS 2
#define NUM_MAX_RDMA_PEERS 20

// [dyogev patch] Byte stride per NVL-island slice in the `is_token_in_rank`
// matrix. The HT dispatch kernel reads each island slice as a uint64_t for
// efficiency (single packed load, fast nonzero test, broadcastable across
// lanes via `broadcast<uint64_t>` which requires sizeof(T) % sizeof(int) == 0).
// Pad to 8 bytes when NUM_MAX_NVL_PEERS < 8 so the reinterpret_cast<uint64_t>
// is naturally aligned and `broadcast<uint64_t>` instantiates cleanly.
// Collapses to NUM_MAX_NVL_PEERS in production (>= 8).
#define IS_TOKEN_IN_RANK_ISLAND_STRIDE ((NUM_MAX_NVL_PEERS) < 8 ? 8 : (NUM_MAX_NVL_PEERS))
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#else
#define NUM_CPU_TIMEOUT_SECS 10
#endif

#define EP_SEND_PHASE 1
#define EP_RECV_PHASE 2

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef DISABLE_SM90_FEATURES
#include <cuda_fp8.h>
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif

#include <infiniband/mlx5dv.h>

namespace nixl_ep {

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif

#define INT_BITS_T2(bits) int##bits##_t
#define INT_BITS_T(bits) INT_BITS_T2(bits)
typedef INT_BITS_T(TOPK_IDX_BITS) topk_idx_t;
#undef INT_BITS_T
#undef INT_BITS_T2

} // namespace nixl_ep
