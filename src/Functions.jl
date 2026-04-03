# =============================================================================
# Functions.jl — Operator registry for GP V2
#
# Operator tiers and index ranges
# ─────────────────────────────────────────────────────────────────
#  Tier 1 — Elementwise (work on a Vector{Float64} or scalar T)
#    UNARY    idx  1 –  9  : safe_sqrt, safe_log, safe_inv, safe_neg,
#                            safe_sin, safe_cos, safe_tan, safe_tanh, abs
#    BINARY   idx 10 – 15  : safe_add, safe_sub, safe_mul, safe_div,
#                            safe_pow, signed_power
#
#  Tier 2 — Financial / contextual (operate on evaluated signal vectors)
#    TS_UNARY  idx 16 – 23  : ts_delta, ts_mean, ts_stddev, ts_rank,
#                             ts_max, ts_min, ts_sum, decay_linear
#    TS_BINARY idx 24 – 25  : ts_corr, ts_cov
#    CS        idx 26 – 29  : cs_rank, cs_zscore, cs_scale, cs_winsorize
#
# Node encoding for financial ops
# ─────────────────────────────────────────────────────────────────
#  TS_UNARY  : node.left     = index of child subtree (any expression)
#              node.right    = 0 (unused)
#              node.const_idx = window length d
#
#  TS_BINARY : node.left     = index of first child subtree
#              node.right    = index of second child subtree
#              node.const_idx = window length d
#
#  CS        : node.left     = index of child subtree
#              node.right    = 0 (unused)
#              node.const_idx = 0 (unused)
#
#  TS functions receive already-evaluated Vector{Float64} signals —
#  NOT (X, col) pairs. This enables composite arguments such as
#  ts_mean(safe_add(x1, x2), 10) without any special-casing in eval.
#
# Adding a new operator
# ─────────────────────────────────────────────────────────────────
#  1. Implement the function in the appropriate section below.
#  2. Append it to the correct *_OPS tuple.
#  3. Add Symbol → Int8 entry to OP_SYMBOL_TO_IDX.
#  4. Add Int8 → String entry to OP_NAMES.
#  Nothing else changes — all downstream dispatch uses the index.
# =============================================================================

using LinearAlgebra
using Base


# =============================================================================
# TIER 1 — ELEMENTWISE OPERATORS
# =============================================================================

# Unary 
@inline safe_log(x::T)  where {T<:Real} = log(Base.abs(x) + T(1e-6))
@inline safe_sqrt(x::T) where {T<:Real} = sqrt(Base.abs(x) + T(1e-6))
@inline safe_inv(x::T)  where {T<:Real} = one(T) / (Base.abs(x) + T(1e-6))
@inline safe_neg(x::T)  where {T<:Real} = -x
@inline safe_sin(x::T)  where {T<:Real} = sin(clamp(x, T(-1e4), T(1e4)))
@inline safe_cos(x::T)  where {T<:Real} = cos(clamp(x, T(-1e4), T(1e4)))
@inline safe_tan(x::T)  where {T<:Real} = tan(clamp(x, T(-1.5), T(1.5)))
@inline safe_tanh(x::T) where {T<:Real} = tanh(x)

# Binary
@inline safe_add(x::T, y::T) where {T<:Real} = x + y
@inline safe_sub(x::T, y::T) where {T<:Real} = x - y
@inline safe_mul(x::T, y::T) where {T<:Real} = x * y
@inline safe_div(x::Real, y::Real) = x / (Base.abs(y) + 1e-6)
@inline safe_pow(x::Real, y::Real) = Base.abs(x)^clamp(y, -5.0, 5.0)
@inline signed_power(x::Real, e::Real) = sign(x) * Base.abs(x)^clamp(e, -5.0, 5.0)

const UNARY_OPS = (safe_sqrt, safe_log, safe_inv, safe_neg,
                   safe_sin, safe_cos, safe_tan, safe_tanh, Base.abs)

const BINARY_OPS = (safe_add, safe_sub, safe_mul, safe_div, safe_pow, signed_power)

const N_UNARY  = Int8(length(UNARY_OPS))
const N_BINARY = Int8(length(BINARY_OPS))
const N_ELEM   = Int8(N_UNARY + N_BINARY)
const N_OPS    = N_ELEM


# =============================================================================
# TIER 2 — FINANCIAL OPERATORS
# =============================================================================
#
# All TS functions receive a pre-evaluated Vector{Float64} signal (not X + col).
# This is what enables composite arguments: the child subtree is evaluated
# first by eval_node, and the resulting vector is passed directly here.
#
# TS_UNARY  signature: (v::Vector{Float64}, d::Int) → Vector{Float64}
# TS_BINARY signature: (v1::Vector{Float64}, v2::Vector{Float64}, d::Int) → Vector{Float64}
# CS        signature: (v::Vector{Float64}) → Vector{Float64}   (unchanged)
# =============================================================================

# -----------------------------------------------------------------------------
# TS_UNARY — one signal vector + window d   
# -----------------------------------------------------------------------------

"""Change over d periods: v[i] - v[i-d]. First d rows are 0."""
function ts_delta(v::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v)
    out = zeros(Float64, n)
    @inbounds for i in (d + 1):n
        out[i] = v[i] - v[i - d]
    end
    out
end

"""Rolling arithmetic mean over the past d rows."""
function ts_mean(v::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v)
    out = zeros(Float64, n)
    @inbounds for i in d:n
        s = 0.0
        for j in (i - d + 1):i; s += v[j]; end
        out[i] = s / d
    end
    out
end

"""Rolling sample standard deviation over the past d rows."""
function ts_stddev(v::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v)
    out = zeros(Float64, n)
    d < 2 && return out
    @inbounds for i in d:n
        μ = 0.0
        for j in (i - d + 1):i; μ += v[j]; end
        μ /= d
        var = 0.0
        for j in (i - d + 1):i; var += (v[j] - μ)^2; end
        var /= (d - 1)
        out[i] = var < 1e-12 ? 0.0 : sqrt(var)
    end
    out
end

"""Rolling percentile rank: fraction of past d values ≤ current. Output in [0,1]."""
function ts_rank(v::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v)
    out = zeros(Float64, n)
    @inbounds for i in d:n
        cur = v[i]
        cnt = 0
        for j in (i - d + 1):i; cnt += (v[j] <= cur); end
        out[i] = cnt / d
    end
    out
end

"""Rolling maximum over the past d rows."""
function ts_max(v::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v)
    out = zeros(Float64, n)
    @inbounds for i in d:n
        m = v[i - d + 1]
        for j in (i - d + 2):i
            vj = v[j]; m = vj > m ? vj : m
        end
        out[i] = m
    end
    out
end

"""Rolling minimum over the past d rows."""
function ts_min(v::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v)
    out = zeros(Float64, n)
    @inbounds for i in d:n
        m = v[i - d + 1]
        for j in (i - d + 2):i
            vj = v[j]; m = vj < m ? vj : m
        end
        out[i] = m
    end
    out
end

"""Rolling sum over the past d rows."""
function ts_sum(v::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v)
    out = zeros(Float64, n)
    @inbounds for i in d:n
        s = 0.0
        for j in (i - d + 1):i; s += v[j]; end
        out[i] = s
    end
    out
end

"""Linearly weighted moving average. Weights 1…d (newest = d), normalised."""
function decay(v::Vector{Float64}, d::Int)::Vector{Float64}
    n     = length(v)
    out   = zeros(Float64, n)
    denom = d * (d + 1) / 2
    @inbounds for i in d:n
        s = 0.0
        for k in 1:d; s += k * v[i - d + k]; end
        out[i] = s / denom
    end
    out
end


# -----------------------------------------------------------------------------
# TS_BINARY — two signal vectors + window d
# -----------------------------------------------------------------------------

"""Rolling Pearson correlation between two signal vectors over d rows."""
function ts_corr(v1::Vector{Float64}, v2::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v1)
    out = zeros(Float64, n)
    d < 2 && return out
    @inbounds for i in d:n
        mx = 0.0; my = 0.0
        for j in (i - d + 1):i; mx += v1[j]; my += v2[j]; end
        mx /= d;  my /= d
        num = 0.0; sx = 0.0; sy = 0.0
        for j in (i - d + 1):i
            dx = v1[j] - mx; dy = v2[j] - my
            num += dx * dy; sx += dx^2; sy += dy^2
        end
        dn = sqrt(sx * sy)
        out[i] = dn < 1e-8 ? 0.0 : num / dn
    end
    out
end

"""Rolling sample covariance between two signal vectors over d rows."""
function ts_cov(v1::Vector{Float64}, v2::Vector{Float64}, d::Int)::Vector{Float64}
    n   = length(v1)
    out = zeros(Float64, n)
    d < 2 && return out
    @inbounds for i in d:n
        mx = 0.0; my = 0.0
        for j in (i - d + 1):i; mx += v1[j]; my += v2[j]; end
        mx /= d; my /= d
        cov = 0.0
        for j in (i - d + 1):i
            cov += (v1[j] - mx) * (v2[j] - my)
        end
        out[i] = cov / (d - 1)
    end
    out
end


# -----------------------------------------------------------------------------
# CS_OPS — cross-sectional: operate on evaluated Vector{Float64} (unchanged)
# -----------------------------------------------------------------------------

"""Fractional rank across all elements. Output in (0, 1]."""
function cs_rank(v::Vector{Float64})::Vector{Float64}
    n = length(v); n == 0 && return v
    idx = sortperm(v); out = similar(v)
    @inbounds for (rank, i) in enumerate(idx); out[i] = rank / n; end
    out
end

"""Standardise: (v - mean) / std. Returns zeros if std < 1e-8."""
function cs_zscore(v::Vector{Float64})::Vector{Float64}
    μ = sum(v) / length(v)
    σ = sqrt(sum((x - μ)^2 for x in v) / max(length(v) - 1, 1))
    σ < 1e-8 && return zeros(length(v))
    (v .- μ) ./ σ
end

"""Scale so sum(abs(v)) == 1. Returns copy of v if near-zero."""
function cs_scale(v::Vector{Float64})::Vector{Float64}
    s = sum(Base.abs, v)
    s < 1e-8 ? copy(v) : v ./ s
end

"""Clip outliers at ±3 standard deviations from the mean."""
function cs_winsorize(v::Vector{Float64})::Vector{Float64}
    μ = sum(v) / length(v)
    σ = sqrt(sum((x - μ)^2 for x in v) / max(length(v) - 1, 1))
    σ < 1e-8 && return copy(v)
    clamp.(v, μ - 3σ, μ + 3σ)
end


# =============================================================================
# TIER 2 REGISTRY
# =============================================================================

const TS_UNARY_OPS = (ts_delta, ts_mean, ts_stddev, ts_rank,
                      ts_max, ts_min, ts_sum, decay)

const TS_BINARY_OPS = (ts_corr, ts_cov)

const CS_OPS = (cs_rank, cs_zscore, cs_scale, cs_winsorize)

const N_TS_UNARY  = Int8(length(TS_UNARY_OPS))
const N_TS_BINARY = Int8(length(TS_BINARY_OPS))
const N_CS        = Int8(length(CS_OPS))

const IDX_TS_UNARY_START  = Int8(N_ELEM + 1)
const IDX_TS_UNARY_END    = Int8(N_ELEM + N_TS_UNARY)
const IDX_TS_BINARY_START = Int8(N_ELEM + N_TS_UNARY + 1)
const IDX_TS_BINARY_END   = Int8(N_ELEM + N_TS_UNARY + N_TS_BINARY)
const IDX_CS_START        = Int8(N_ELEM + N_TS_UNARY + N_TS_BINARY + 1)
const IDX_CS_END          = Int8(N_ELEM + N_TS_UNARY + N_TS_BINARY + N_CS)
const N_ALL_OPS           = IDX_CS_END


# =============================================================================
# HUMAN-READABLE NAMES
# =============================================================================

const OP_NAMES = Dict{Int8, String}(
    Int8(1)  => "safe_sqrt",   Int8(2)  => "safe_log",
    Int8(3)  => "safe_inv",    Int8(4)  => "safe_neg",
    Int8(5)  => "safe_sin",    Int8(6)  => "safe_cos",
    Int8(7)  => "safe_tan",    Int8(8)  => "safe_tanh",
    Int8(9)  => "abs",
    Int8(10) => "safe_add",    Int8(11) => "safe_sub",
    Int8(12) => "safe_mul",    Int8(13) => "safe_div",
    Int8(14) => "safe_pow",    Int8(15) => "signed_power",
    Int8(16) => "ts_delta",    Int8(17) => "ts_mean",
    Int8(18) => "ts_stddev",   Int8(19) => "ts_rank",
    Int8(20) => "ts_max",      Int8(21) => "ts_min",
    Int8(22) => "ts_sum",      Int8(23) => "decay",
    Int8(24) => "ts_corr",     Int8(25) => "ts_cov",
    Int8(26) => "cs_rank",     Int8(27) => "cs_zscore",
    Int8(28) => "cs_scale",    Int8(29) => "cs_winsorize",
)

const OP_SYMBOL_TO_IDX = Dict{Symbol, Int8}(
    :safe_sqrt => Int8(1),   :safe_log  => Int8(2),
    :safe_inv  => Int8(3),   :safe_neg  => Int8(4),
    :safe_sin  => Int8(5),   :safe_cos  => Int8(6),
    :safe_tan  => Int8(7),   :safe_tanh => Int8(8),
    :abs       => Int8(9),
    :safe_add  => Int8(10),  :safe_sub  => Int8(11),
    :safe_mul  => Int8(12),  :safe_div  => Int8(13),
    :safe_pow  => Int8(14),  :signed_power => Int8(15),
    :ts_delta  => Int8(16),  :ts_mean   => Int8(17),
    :ts_stddev => Int8(18),  :ts_rank   => Int8(19),
    :ts_max    => Int8(20),  :ts_min    => Int8(21),
    :ts_sum    => Int8(22),  :decay     => Int8(23),
    :ts_corr   => Int8(24),  :ts_cov    => Int8(25),
    :cs_rank   => Int8(26),  :cs_zscore => Int8(27),
    :cs_scale  => Int8(28),  :cs_winsorize => Int8(29),
)


# =============================================================================
# PREDICATES
# =============================================================================

@inline is_unary_op(op_idx::Int8)     = Int8(1) <= op_idx <= N_UNARY
@inline is_binary_op(op_idx::Int8)    = N_UNARY < op_idx <= N_ELEM
@inline is_ts_unary_op(op_idx::Int8)  = IDX_TS_UNARY_START  <= op_idx <= IDX_TS_UNARY_END
@inline is_ts_binary_op(op_idx::Int8) = IDX_TS_BINARY_START <= op_idx <= IDX_TS_BINARY_END
@inline is_cs_op(op_idx::Int8)        = IDX_CS_START        <= op_idx <= IDX_CS_END
@inline is_elem_op(op_idx::Int8)      = Int8(1) <= op_idx <= N_ELEM
@inline is_financial_op(op_idx::Int8) = op_idx > N_ELEM

@inline function get_arity(op_idx::Int8)::Int
    is_unary_op(op_idx)     && return 1
    is_binary_op(op_idx)    && return 2
    is_ts_unary_op(op_idx)  && return 1
    is_ts_binary_op(op_idx) && return 2
    is_cs_op(op_idx)        && return 1
    error("get_arity: invalid op_idx=$op_idx (valid range 1:$N_ALL_OPS)")
end

@inline function op_name(op_idx::Int8)::String
    haskey(OP_NAMES, op_idx) && return OP_NAMES[op_idx]
    error("op_name: invalid op_idx=$op_idx")
end


# =============================================================================
# TIER 1 DISPATCH
# =============================================================================

@inline function apply_unary_scalar(op_idx::Int8, x::T)::T where {T<:Real}
    @boundscheck is_unary_op(op_idx) ||
        throw(BoundsError("apply_unary_scalar: op_idx=$op_idx out of unary range [1,$N_UNARY]"))
    @inbounds UNARY_OPS[op_idx](x)
end

@inline function apply_binary_scalar(op_idx::Int8, x::T, y::T)::T where {T<:Real}
    local_idx = op_idx - N_UNARY
    @boundscheck (Int8(1) <= local_idx <= N_BINARY) ||
        throw(BoundsError("apply_binary_scalar: op_idx=$op_idx out of binary range"))
    @inbounds BINARY_OPS[local_idx](x, y)
end

function apply_unary_op(op_idx::Int8, x::Vector{T})::Vector{T} where {T<:Real}
    @boundscheck is_unary_op(op_idx) ||
        throw(BoundsError("apply_unary_op: op_idx=$op_idx out of unary range"))
    @inbounds UNARY_OPS[op_idx].(x)
end

function apply_binary_op(op_idx::Int8, x::Vector{T}, y::Vector{T})::Vector{T} where {T<:Real}
    local_idx = op_idx - N_UNARY
    @boundscheck (Int8(1) <= local_idx <= N_BINARY) ||
        throw(BoundsError("apply_binary_op: op_idx=$op_idx out of binary range"))
    @inbounds BINARY_OPS[local_idx].(x, y)
end


# =============================================================================
# TIER 2 DISPATCH
# =============================================================================

"""
    apply_ts_unary_op(op_idx, v, d) → Vector{Float64}

Apply TS-unary operator to a pre-evaluated signal vector `v` with window `d`.
`v` is the output of evaluating the child subtree — it may be any expression,
not just a raw feature column.
"""
function apply_ts_unary_op(op_idx::Int8, v::Vector{Float64}, d::Int)::Vector{Float64}
    local_idx = op_idx - N_ELEM
    @boundscheck (1 <= local_idx <= N_TS_UNARY) ||
        throw(BoundsError("apply_ts_unary_op: op_idx=$op_idx out of TS-unary range"))
    @inbounds TS_UNARY_OPS[local_idx](v, d)
end

"""
    apply_ts_binary_op(op_idx, v1, v2, d) → Vector{Float64}

Apply TS-binary operator to two pre-evaluated signal vectors with window `d`.
"""
function apply_ts_binary_op(op_idx::Int8, v1::Vector{Float64},
                             v2::Vector{Float64}, d::Int)::Vector{Float64}
    local_idx = op_idx - N_ELEM - N_TS_UNARY
    @boundscheck (1 <= local_idx <= N_TS_BINARY) ||
        throw(BoundsError("apply_ts_binary_op: op_idx=$op_idx out of TS-binary range"))
    @inbounds TS_BINARY_OPS[local_idx](v1, v2, d)
end

"""
    apply_cs_op(op_idx, v) → Vector{Float64}

Apply a cross-sectional operator on an already-evaluated signal vector.
"""
function apply_cs_op(op_idx::Int8, v::Vector{Float64})::Vector{Float64}
    local_idx = op_idx - N_ELEM - N_TS_UNARY - N_TS_BINARY
    @boundscheck (1 <= local_idx <= N_CS) ||
        throw(BoundsError("apply_cs_op: op_idx=$op_idx out of CS range"))
    @inbounds CS_OPS[local_idx](v)
end


# =============================================================================
# STARTUP HELPERS
# =============================================================================

function Symbol_to_op_idx(name::Symbol)::Int8
    haskey(OP_SYMBOL_TO_IDX, name) && return OP_SYMBOL_TO_IDX[name]
    error("Symbol_to_op_idx: ':$name' is not a registered operator.\n" *
          "Registered: $(sort(collect(keys(OP_SYMBOL_TO_IDX))))")
end

function validate_op_symbols(config)
    unary     = [Symbol_to_op_idx(s) for s in config.unary_op_names]
    binary    = [Symbol_to_op_idx(s) for s in config.binary_op_names]
    ts_unary  = [Symbol_to_op_idx(s) for s in config.ts_op_names]
    ts_binary = [Symbol_to_op_idx(s) for s in config.ts_binary_op_names]
    cs        = [Symbol_to_op_idx(s) for s in config.cs_op_names]
    return unary, binary, ts_unary, ts_binary, cs
end