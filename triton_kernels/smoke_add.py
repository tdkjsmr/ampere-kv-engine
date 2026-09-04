"""用于 G0 环境验证的最小 Triton 向量加法。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _smoke_add_kernel(
    left_ptr,
    right_ptr,
    output_ptr,
    num_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    """每个 Triton Program 处理一段连续元素。"""

    # program_id 表示当前 Program 在一维 Grid 中的编号。
    program_id = tl.program_id(axis=0)
    offsets = program_id * block_size + tl.arange(0, block_size)

    # 最后一个 Program 可能越过向量末尾，所有访存都必须使用同一 Mask。
    mask = offsets < num_elements
    left = tl.load(left_ptr + offsets, mask=mask)
    right = tl.load(right_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, left + right, mask=mask)


def triton_smoke_add(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """启动 Triton Smoke Kernel，并返回与输入同形状的 Tensor。"""

    if not left.is_cuda or not right.is_cuda:
        raise ValueError("Triton Smoke 的两个输入都必须位于 CUDA Device")
    if left.device != right.device:
        raise ValueError("Triton Smoke 的两个输入必须位于同一 CUDA Device")
    if left.shape != right.shape:
        raise ValueError("Triton Smoke 的两个输入形状必须一致")
    if left.dtype != right.dtype:
        raise ValueError("Triton Smoke 的两个输入 dtype 必须一致")
    if not left.is_contiguous() or not right.is_contiguous():
        raise ValueError("Triton Smoke 只接受连续 Tensor")

    output = torch.empty_like(left)
    num_elements = left.numel()
    if num_elements == 0:
        return output

    block_size = 256
    grid = (triton.cdiv(num_elements, block_size),)
    _smoke_add_kernel[grid](
        left,
        right,
        output,
        num_elements=num_elements,
        block_size=block_size,
    )
    return output
