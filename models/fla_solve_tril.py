"""JAX wrapper for FLA's Triton solve_tril kernel.

Calls a stripped-down version of FLA's merge_16x16_to_64x64_inverse_kernel
via jax_triton.triton_call(). Computes (I + A)^{-1} where A is strictly lower
triangular, using hierarchical 4x16 block inverse (all in SRAM).

Adapted from: flash-linear-attention/fla/ops/utils/solve_tril.py
Changes from FLA original:
  - Removed IS_VARLEN / cu_seqlens / chunk_indices (we use fixed-length sequences)
  - Removed USE_TMA (GH200 sm_90a does not support TMA)
  - Reordered parameters: output Ai is LAST (required by jax_triton.triton_call)
"""
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
import jax_triton as jt


# ---------------------------------------------------------------------------
# Triton kernel: hierarchical 4x16 block inverse for 64x64 matrices
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT'],
)
@triton.jit
def _solve_tril_64x64_kernel(
    # Input (from *args)
    A,
    # Scalar args (from *args, re-inserted by jax_triton)
    T,
    # Output (appended by jax_triton from out_shape)
    Ai,
    # Constexpr (from **metaparams)
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Load 4 diagonal 16x16 blocks
    p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    # Mask to strictly lower triangular and negate
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    # Forward substitution on each 16x16 diagonal block (14 steps each)
    for i in range(2, 16):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, 32):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, 48):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, 64):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    # Load 6 off-diagonal blocks from A
    p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    # Block elimination merge (6 GEMM operations)
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32), b_Ai_22)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43), b_Ai_33)
    b_Ai_31 = -tl.dot(b_Ai_33, tl.dot(b_A_31, b_Ai_11) + tl.dot(b_A_32, b_Ai_21))
    b_Ai_42 = -tl.dot(b_Ai_44, tl.dot(b_A_42, b_Ai_22) + tl.dot(b_A_43, b_Ai_32))
    b_Ai_41 = -tl.dot(b_Ai_44, tl.dot(b_A_41, b_Ai_11) + tl.dot(b_A_42, b_Ai_21) + tl.dot(b_A_43, b_Ai_31))

    # Store all 16 blocks to Ai (10 computed + 6 upper-triangle zeros)
    b_zero = tl.zeros([16, 16], dtype=tl.float32)
    # Row 0: [diag, zero, zero, zero]
    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_12 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 16), (16, 16), (1, 0))
    p_Ai_13 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 32), (16, 16), (1, 0))
    p_Ai_14 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 48), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_12, b_zero.to(p_Ai_12.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_13, b_zero.to(p_Ai_13.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_14, b_zero.to(p_Ai_14.dtype.element_ty), boundary_check=(0, 1))
    # Row 1: [off-diag, diag, zero, zero]
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_23 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 32), (16, 16), (1, 0))
    p_Ai_24 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 48), (16, 16), (1, 0))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_23, b_zero.to(p_Ai_23.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_24, b_zero.to(p_Ai_24.dtype.element_ty), boundary_check=(0, 1))
    # Row 2: [off-diag, off-diag, diag, zero]
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_34 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 48), (16, 16), (1, 0))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_34, b_zero.to(p_Ai_34.dtype.element_ty), boundary_check=(0, 1))
    # Row 3: [off-diag, off-diag, off-diag, diag]
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# JAX wrapper
# ---------------------------------------------------------------------------
def solve_tril_64(A):
    """Compute (I + A)^{-1} for batched strictly lower triangular 64x64 matrices.

    Uses FLA's hierarchical block inverse algorithm via Triton kernel.

    Args:
        A: [B, T, H, 64] float32 -- strictly lower triangular in FLA layout.
            B = batch, T = num_chunks * 64, H = heads, BT = 64.

    Returns:
        Ai: [B, T, H, 64] float32 -- (I + A)^{-1}, same layout.
    """
    B, T, H, BT = A.shape
    assert BT == 64, f"solve_tril_64 requires BT=64, got {BT}"
    NT = T // BT

    # triton_call convention:
    #   positional *args = [A (array), T (scalar)] -> mapped to kernel params A, T
    #   out_shape -> appended as kernel param Ai
    #   **metaparams -> constexpr params H, BT
    Ai = jt.triton_call(
        A,       # -> kernel param: A (pointer)
        T,       # -> kernel param: T (int32 scalar)
        kernel=_solve_tril_64x64_kernel,
        out_shape=jax.ShapeDtypeStruct(A.shape, jnp.float32),
        grid=(NT, B * H),
        # constexpr metaparams
        H=H,
        BT=BT,
    )
    # Kernel writes all 16 blocks (10 computed + 6 upper-tri zeros in-kernel).
    return Ai
