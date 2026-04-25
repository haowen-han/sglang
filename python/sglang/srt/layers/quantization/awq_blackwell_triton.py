# SPDX-License-Identifier: Apache-2.0
"""Blackwell AWQ GEMM kernel via Triton.

Weight format (Blackwell path):
  - b_qweight: [K, N/8] int32, stride (1, K), AWQ interleave packing [0,2,4,6,1,3,5,7]
  - scales:    [G, N]    bf16, stride (1, G), group-contiguous (G = K / group_size)
  - zeros:     [G, N/8]  int32, stride (1, G), AWQ interleave packing
  - a:         [M, K]    bf16, stride (K, 1)
  - output:    [M, N]    bf16
"""

import torch
import triton
import triton.language as tl


@triton.jit
def blackwell_awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    group_size,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_zg: tl.constexpr,
    stride_zn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # TODO: implement
    pass


def blackwell_awq_gemm(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
) -> torch.Tensor:
    """Blackwell AWQ GEMM: C = A @ dequant(B).

    Args:
        a: [M, K] bf16 activation matrix, stride (K, 1).
        b_qweight: [K, N/8] int32 packed weights, stride (1, K),
            AWQ interleave packing (bit 0-3 = channel 0, bit 4-7 = channel 2, ...).
        scales: [G, N] bf16 quantization scales, stride (1, G),
            where G = K / group_size.
        zeros: [G, N/8] int32 packed zero-points, stride (1, G),
            AWQ interleave packing, same format as b_qweight.
        M: number of rows in A.
        N: number of columns in output (before packing).
        K: number of columns in A / rows in B.
        group_size: number of K rows per quantization group.

    Returns:
        [M, N] bf16 output matrix.
    """
    # TODO: implement kernel launch
    raise NotImplementedError("blackwell_awq_gemm kernel not yet implemented")
