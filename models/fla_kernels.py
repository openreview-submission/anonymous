"""Extracted FLA Triton kernels for GDN — zero torch dependency.

Provides two operations:
  1. solve_tril_64: (I + A)^{-1} via hierarchical 16x16 block-inverse
  2. wy_fwd: w = A @ (k * beta * exp(g)), u = A @ (v * beta)

Extracted from flash-linear-attention (Copyright 2023-2025 Songlin Yang, Yu Zhang).
Modifications:
  - Removed torch/fla imports, inlined constants for GH200 (sm_90a)
  - Reordered kernel params: [inputs, scalars, outputs, constexpr]
    (required by jax_triton.triton_call which appends out_shape buffers last)
"""
import inspect
import jax
import jax.numpy as jnp
import triton
import triton.language as tl
import jax_triton as jt

# ---------------------------------------------------------------------------
# Inlined FLA constants for GH200 (sm_90a, CC 9.0)
# ---------------------------------------------------------------------------
_SUPPORTS_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
_CACHE_KWARGS = {"cache_results": True} if _SUPPORTS_CACHE else {}


# ---------------------------------------------------------------------------
# Inlined FLA helpers (pure triton, no torch)
# ---------------------------------------------------------------------------
@triton.jit
def _exp(x):
    return tl.exp(x.to(tl.float32))


# ============================================================================
# Kernel 1: solve_tril 64x64
# ============================================================================
# Full 64x64 block-inverse: 4 diagonal 16x16 forward-substitution +
# 6 off-diagonal Schur complement merges. All in SRAM.
#
# From: fla/ops/utils/solve_tril.py (merge_16x16_to_64x64_inverse_kernel)
# Simplified: removed IS_VARLEN, USE_TMA paths (fixed-length LOB data)
#
# Param order: [A(input), T(scalar), Ai(output), H/BT/DOT_PRECISION(constexpr)]
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': 'ieee'}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'BT'],
    **_CACHE_KWARGS,
)
@triton.jit(do_not_specialize=['T'])
def _solve_tril_64x64_kernel(
    A,      # input: strictly lower triangular
    T,      # scalar: total sequence length (not specialized)
    Ai,     # output: (I+A)^{-1} (from out_shape)
    H: tl.constexpr,
    BT: tl.constexpr,  # must be 64
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    base_A = A + (bos * H + i_h) * BT
    base_Ai = Ai + (bos * H + i_h) * BT
    stride_t = H * BT

    o_i = tl.arange(0, 16)
    m_lower = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    # --- Load and invert 4 diagonal 16x16 blocks ---
    offs_00 = (i_t * BT + o_i[:, None]) * stride_t + o_i[None, :]
    b_A00 = tl.load(base_A + offs_00).to(tl.float32)
    b_A00 = tl.where(m_lower, b_A00, 0.)
    for i in range(1, 16):
        mask = o_i == i
        b_a = tl.sum(tl.where(mask[:, None], b_A00, 0.), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A00, 0)
        b_A00 = tl.where(mask[:, None], b_a[None, :], b_A00)
    b_A00 += m_I

    offs_11 = (i_t * BT + 16 + o_i[:, None]) * stride_t + 16 + o_i[None, :]
    b_A11 = tl.load(base_A + offs_11).to(tl.float32)
    b_A11 = tl.where(m_lower, b_A11, 0.)
    for i in range(1, 16):
        mask = o_i == i
        b_a = tl.sum(tl.where(mask[:, None], b_A11, 0.), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A11, 0)
        b_A11 = tl.where(mask[:, None], b_a[None, :], b_A11)
    b_A11 += m_I

    offs_22 = (i_t * BT + 32 + o_i[:, None]) * stride_t + 32 + o_i[None, :]
    b_A22 = tl.load(base_A + offs_22).to(tl.float32)
    b_A22 = tl.where(m_lower, b_A22, 0.)
    for i in range(1, 16):
        mask = o_i == i
        b_a = tl.sum(tl.where(mask[:, None], b_A22, 0.), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A22, 0)
        b_A22 = tl.where(mask[:, None], b_a[None, :], b_A22)
    b_A22 += m_I

    offs_33 = (i_t * BT + 48 + o_i[:, None]) * stride_t + 48 + o_i[None, :]
    b_A33 = tl.load(base_A + offs_33).to(tl.float32)
    b_A33 = tl.where(m_lower, b_A33, 0.)
    for i in range(1, 16):
        mask = o_i == i
        b_a = tl.sum(tl.where(mask[:, None], b_A33, 0.), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A33, 0)
        b_A33 = tl.where(mask[:, None], b_a[None, :], b_A33)
    b_A33 += m_I

    # --- Load 6 off-diagonal blocks ---
    offs_10 = (i_t * BT + 16 + o_i[:, None]) * stride_t + o_i[None, :]
    offs_20 = (i_t * BT + 32 + o_i[:, None]) * stride_t + o_i[None, :]
    offs_21 = (i_t * BT + 32 + o_i[:, None]) * stride_t + 16 + o_i[None, :]
    offs_30 = (i_t * BT + 48 + o_i[:, None]) * stride_t + o_i[None, :]
    offs_31 = (i_t * BT + 48 + o_i[:, None]) * stride_t + 16 + o_i[None, :]
    offs_32 = (i_t * BT + 48 + o_i[:, None]) * stride_t + 32 + o_i[None, :]

    b_L10 = tl.load(base_A + offs_10).to(tl.float32)
    b_L20 = tl.load(base_A + offs_20).to(tl.float32)
    b_L21 = tl.load(base_A + offs_21).to(tl.float32)
    b_L30 = tl.load(base_A + offs_30).to(tl.float32)
    b_L31 = tl.load(base_A + offs_31).to(tl.float32)
    b_L32 = tl.load(base_A + offs_32).to(tl.float32)

    # --- Merge: Schur complement block elimination ---
    # Block inverse of M = I - A (lower triangular):
    #   M^{-1}[i,j] = -M_ii^{-1} @ M_ij @ M_jj^{-1}
    # Since M_ij = -A_ij (off-diagonal), the two negatives cancel:
    #   result[i,j] = Ai_ii @ A_ij @ Ai_jj  (POSITIVE)
    Ai_10 = tl.dot(tl.dot(b_A11, b_L10, input_precision=DOT_PRECISION),
                    b_A00, input_precision=DOT_PRECISION)
    Ai_21 = tl.dot(tl.dot(b_A22, b_L21, input_precision=DOT_PRECISION),
                    b_A11, input_precision=DOT_PRECISION)
    Ai_32 = tl.dot(tl.dot(b_A33, b_L32, input_precision=DOT_PRECISION),
                    b_A22, input_precision=DOT_PRECISION)
    Ai_20 = tl.dot(b_A22,
                    tl.dot(b_L20, b_A00, input_precision=DOT_PRECISION) +
                    tl.dot(b_L21, Ai_10, input_precision=DOT_PRECISION),
                    input_precision=DOT_PRECISION)
    Ai_31 = tl.dot(b_A33,
                    tl.dot(b_L31, b_A11, input_precision=DOT_PRECISION) +
                    tl.dot(b_L32, Ai_21, input_precision=DOT_PRECISION),
                    input_precision=DOT_PRECISION)
    Ai_30 = tl.dot(b_A33,
                    tl.dot(b_L30, b_A00, input_precision=DOT_PRECISION) +
                    tl.dot(b_L31, Ai_10, input_precision=DOT_PRECISION) +
                    tl.dot(b_L32, Ai_20, input_precision=DOT_PRECISION),
                    input_precision=DOT_PRECISION)

    # --- Store all 16 blocks (10 lower + 6 upper zeros) ---
    tl.store(base_Ai + offs_00, b_A00.to(tl.float32))
    tl.store(base_Ai + offs_10, Ai_10.to(tl.float32))
    tl.store(base_Ai + offs_11, b_A11.to(tl.float32))
    tl.store(base_Ai + offs_20, Ai_20.to(tl.float32))
    tl.store(base_Ai + offs_21, Ai_21.to(tl.float32))
    tl.store(base_Ai + offs_22, b_A22.to(tl.float32))
    tl.store(base_Ai + offs_30, Ai_30.to(tl.float32))
    tl.store(base_Ai + offs_31, Ai_31.to(tl.float32))
    tl.store(base_Ai + offs_32, Ai_32.to(tl.float32))
    tl.store(base_Ai + offs_33, b_A33.to(tl.float32))

    z = tl.zeros([16, 16], dtype=tl.float32)
    tl.store(base_Ai + (i_t * BT + o_i[:, None]) * stride_t + 16 + o_i[None, :], z)
    tl.store(base_Ai + (i_t * BT + o_i[:, None]) * stride_t + 32 + o_i[None, :], z)
    tl.store(base_Ai + (i_t * BT + o_i[:, None]) * stride_t + 48 + o_i[None, :], z)
    tl.store(base_Ai + (i_t * BT + 16 + o_i[:, None]) * stride_t + 32 + o_i[None, :], z)
    tl.store(base_Ai + (i_t * BT + 16 + o_i[:, None]) * stride_t + 48 + o_i[None, :], z)
    tl.store(base_Ai + (i_t * BT + 32 + o_i[:, None]) * stride_t + 48 + o_i[None, :], z)


# ============================================================================
# Kernel 2: recompute_w_u_fwd (GDN scalar gate version)
# ============================================================================
# From: fla/ops/gated_delta_rule/wy_fast.py
# Simplified: removed IS_VARLEN path, hardcoded safe_dot = tl.dot
#
# Param order for jt.triton_call:
#   positional arrays: k, v, beta, A, g  (5 inputs)
#   positional scalar: T                  (1 scalar at flat_args[5])
#   out_shape outputs: w, u              (2 outputs, appended)
#   constexpr metaparams: H, K, V, BT, BK, BV, USE_G
#
# jt.triton_call builds: [k, v, beta, A, g] → insert T at pos 5 →
#   [k, v, beta, A, g, T] → append [w, u] → [k, v, beta, A, g, T, w, u]
#   → zip with kernel.arg_names
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G'],
    **_CACHE_KWARGS,
)
@triton.jit(do_not_specialize=['T'])
def _recompute_w_u_fwd_kernel(
    # --- inputs (from positional *args) ---
    k,       # (B, T, H, K)
    v,       # (B, T, H, V)
    beta,    # (B, T, H)
    A,       # (I+L)^{-1}, layout (T, H, BT) — lower triangular
    g,       # (B, T, H) — scalar gate (GDN); dummy array when USE_G=False
    # --- scalar (inserted at original position) ---
    T,
    # --- outputs (from out_shape, appended after inputs+scalars) ---
    w,       # (B, T, H, K)
    u,       # (B, T, H, V)
    # --- constexpr (from **metaparams) ---
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    # Load beta for this chunk
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    # Load A (the triangular inverse) for this chunk
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
                             (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    # u = A @ (v * beta) — loop over V dimension in BV-sized tiles
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                 (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1),
                                 (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    # Load gate if GDN mode
    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_g_last = tl.load(g + (bos + min(i_t * BT + BT, T) - 1) * H + i_h).to(tl.float32)
        b_g_exp = _exp(b_g_last - b_g)  # cumulative decay within chunk

    # w = A @ (k * beta [* exp(g)]) — loop over K dimension in BK-sized tiles
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                 (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1),
                                 (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb = b_kb * b_g_exp[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# JAX Wrappers
# ============================================================================

def solve_tril_64(A_mat):
    """Compute (I - A)^{-1} for batched 64x64 strictly-lower-triangular A.

    FLA convention: computes (I - A)^{-1} via Neumann series I + A + A^2 + ...
    To get (I + A)^{-1}, pass -A: solve_tril_64(-A) = (I + A)^{-1}.

    In GDN: L is negative, so solve_tril_64(L) = (I - L)^{-1} = (I + |L|)^{-1}.

    Args:
        A_mat: (batch, 64, 64) float32 -- strictly lower triangular

    Returns:
        Ai: (batch, 64, 64) float32 -- (I - A)^{-1}
    """
    batch, BT, _ = A_mat.shape
    assert BT == 64, f"Expected BT=64, got {BT}"

    # Kernel expects layout (T, H, BT) with H=1, T=batch*BT
    # Element [t, bt] at offset t*BT + bt -> matches (batch*BT, BT) reshape
    A_flat = A_mat.reshape(batch * BT, BT)
    T = batch * BT

    # jt.triton_call mapping:
    #   positional: A_flat (array), T (scalar at pos 1)
    #   out_shape: Ai_flat (appended)
    #   Final: [A_flat, T, Ai_flat] -> kernel(A, T, Ai, H=1, BT=64, DOT_PRECISION=auto)
    Ai_flat = jt.triton_call(
        A_flat,
        T,
        kernel=_solve_tril_64x64_kernel,
        out_shape=jax.ShapeDtypeStruct(A_flat.shape, jnp.float32),
        grid=(batch, 1),  # (NT=batch, B*H=1)
        H=1,
        BT=BT,
    )
    return Ai_flat.reshape(batch, BT, BT)


def wy_fwd(k, v, beta, A, g=None):
    """Compute w = A @ (k*beta*[exp(g)]), u = A @ (v*beta) via Triton.

    Fuses the matmul with beta scaling and optional gating into a single kernel.

    Args:
        k:    (nc, C, nh, hd)  -- keys
        v:    (nc, C, nh, hvd) -- values
        beta: (nc, C, nh)      -- write gate
        A:    (nc*nh, C, C)    -- (I+L)^{-1} from solve_tril_64
        g:    (nc, C, nh) or None -- scalar decay gate (GDN)

    Returns:
        w: (nc, C, nh, hd)
        u: (nc, C, nh, hvd)
    """
    nc, C, nh, hd = k.shape
    hvd = v.shape[-1]
    BT = C
    BK = min(64, hd)
    BV = min(64, hvd)

    # Reshape to FLA layout: (B=1, T=nc*C, H=nh, D)
    T = nc * C
    k_flat = k.reshape(1, T, nh, hd)
    v_flat = v.reshape(1, T, nh, hvd)
    beta_flat = beta.reshape(1, T, nh)

    # A layout: (nc*nh, C, C) -> need (T, H, BT) = (nc*C, nh, C)
    # (nc, nh, C, C) -> transpose -> (nc, C, nh, C) -> reshape -> (nc*C, nh, C)
    A_for_kernel = A.reshape(nc, nh, C, C).transpose(0, 2, 1, 3).reshape(T, nh, C)

    # Gate: reshape to (1, T, nh) if provided, else dummy (never read)
    use_g = g is not None
    if use_g:
        g_flat = g.reshape(1, T, nh)
    else:
        g_flat = jnp.zeros((1, 1, 1), dtype=jnp.float32)  # dummy, never dereferenced

    NT = nc
    grid = (NT, nh)  # (num_chunks, B*H = 1*nh)

    # jt.triton_call mapping:
    #   positional: k_flat, v_flat, beta_flat, A_for_kernel, g_flat (5 arrays), T (scalar at pos 5)
    #   out_shape: w_out, u_out (2 outputs, appended)
    #   Final: [k, v, beta, A, g, T, w, u] -> matches kernel param order
    w_out, u_out = jt.triton_call(
        k_flat, v_flat, beta_flat, A_for_kernel, g_flat,
        T,  # scalar at position 5 in flat_args
        kernel=_recompute_w_u_fwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((1, T, nh, hd), k_flat.dtype),   # w
            jax.ShapeDtypeStruct((1, T, nh, hvd), v_flat.dtype),  # u
        ],
        grid=grid,
        H=nh,
        K=hd,
        V=hvd,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=use_g,
    )
    return w_out.reshape(nc, C, nh, hd), u_out.reshape(nc, C, nh, hvd)
