# =============================================================================
# Constants.jl — Numeric constant optimization via Optim.jl
#
# Every GP tree carries a small Vector{Float64} of numeric constants.
# After structural changes (crossover, mutation) those constants are random
# leftovers. This file tunes them with a gradient-based optimizer so the tree
# makes the best possible use of whatever structure it has before the next
# generation's selection step.
#
# Loss function: PPS (Predictive Power Score)
# ──────────────────────────────────────────────────────────────────────────
# PPS = β·IC + (1-β)·RankIC is the same metric used by the fitness function,
# so constant optimisation and selection are perfectly aligned.
#
# Why NelderMead (not BFGS):
#   PPS includes RankIC which calls tiedrank — a sort-based operation that
#   is non-differentiable. ForwardDiff would produce zero/wrong gradients
#   through it, making BFGS unreliable. NelderMead is gradient-free and
#   works correctly with the full PPS loss.
#
#   NelderMead converges in ~100-200 function evaluations per tree.
#   Since this only runs every `const_opt_every_n_gens` generations on
#   `elite_count` individuals, the runtime cost is acceptable.
#
# When it runs: every `const_opt_every_n_gens` generations, applied only to
#               the top `elite_count` individuals.
#
# Public API
# ──────────────────────────────────────────────────────────────────────────
#   eval_tree_with_consts(tree, X, consts)                          → Vector{T}
#   optimize_constants!(tree, X, y_raw, y_ranked, config)           → Bool
#   optimize_individual_constants!(ind, X, y_raw, y_ranked, config) → Nothing
#   optimize_population_constants!(pop, X, y_raw, y_ranked, config, gen) → Nothing
# =============================================================================

using Optim


# =============================================================================
# 1. Generic tree evaluator (supports Dual numbers for future use)
# =============================================================================

# Helper: extract Float64 primal from any AbstractVector.
# Used at TS/CS boundaries where those ops are Float64-only.
_to_float64(v::Vector{Float64}) = v
_to_float64(v::AbstractVector)  = map(Float64, v)


"""
    eval_node_with_consts(tree, X, consts, idx) → Vector{T}

Recursive tree evaluator generic over the constant type `T<:Real`.
When `T = Float64` this is identical to `eval_node`.
Kept generic to allow future ForwardDiff use on a fully differentiable
loss (e.g. a soft-rank approximation of RankIC).
"""
function eval_node_with_consts(tree::Tree, X::Matrix{Float64},
                                consts::AbstractVector{T}, idx::Int)::Vector where {T<:Real}
    node   = tree.nodes[idx]
    n_rows = size(X, 1)

    if is_variable(node)
        col = X[:, Int(node.feature)]
        return T === Float64 ? col : T.(col)

    elseif is_constant(node)
        return fill(consts[Int(node.const_idx)], n_rows)

    elseif is_unary_op(node.op_idx)
        child = eval_node_with_consts(tree, X, consts, Int(node.left))::Vector
        return UNARY_OPS[node.op_idx].(child)

    elseif is_binary_op(node.op_idx)
        lv = eval_node_with_consts(tree, X, consts, Int(node.left))::Vector
        rv = eval_node_with_consts(tree, X, consts, Int(node.right))::Vector
        local_idx = node.op_idx - N_UNARY
        return BINARY_OPS[local_idx].(lv, rv)

    elseif is_ts_unary_op(node.op_idx)
        child  = eval_node_with_consts(tree, X, consts, Int(node.left))::Vector
        d      = Int(node.const_idx)
        result = apply_ts_unary_op(node.op_idx, _to_float64(child), d)
        return T === Float64 ? result : T.(result)

    elseif is_ts_binary_op(node.op_idx)
        lv     = eval_node_with_consts(tree, X, consts, Int(node.left))::Vector
        rv     = eval_node_with_consts(tree, X, consts, Int(node.right))::Vector
        d      = Int(node.const_idx)
        result = apply_ts_binary_op(node.op_idx, _to_float64(lv), _to_float64(rv), d)
        return T === Float64 ? result : T.(result)

    elseif is_cs_op(node.op_idx)
        child  = eval_node_with_consts(tree, X, consts, Int(node.left))::Vector
        result = apply_cs_op(node.op_idx, _to_float64(child))
        return T === Float64 ? result : T.(result)
    end

    error("eval_node_with_consts: unhandled node at idx=$idx, op_idx=$(node.op_idx)")
end


"""
    eval_tree_with_consts(tree, X, consts) → Vector{T}

Evaluate `tree` on `X` using `consts` in place of `tree.constants`.
Returns a zero vector for empty trees.
"""
function eval_tree_with_consts(tree::Tree, X::Matrix{Float64},
                                consts::AbstractVector{T})::Vector where {T<:Real}
    isempty(tree.nodes) && return zeros(T, size(X, 1))
    return eval_node_with_consts(tree, X, consts, 1)
end


# =============================================================================
# 2. Single-tree optimizer
# =============================================================================

"""
    optimize_constants!(tree, X, y_raw, y_ranked, config) → Bool

Tune `tree.constants` to maximise PPS on `(X, y_raw, y_ranked)` using
NelderMead (gradient-free).

PPS = β·IC + (1-β)·RankIC is the same metric the fitness function uses,
keeping constant optimisation and selection fully aligned.

Returns `true` when the optimizer ran; `false` when the tree has no
constants. Original constants are preserved on any error.
"""
function optimize_constants!(tree::Tree,
                              X::Matrix{Float64},
                              y_raw::Vector{Float64},
                              y_ranked::Vector{Float64},
                              config::GPConfig)::Bool
    isempty(tree.constants) && return false

    consts0 = extract_constants(tree)

    # Loss: minimise negative PPS — same metric as fitness evaluation
    function loss(c::Vector{Float64})
        sig = eval_tree_with_consts(tree, X, c)
        return -pps(sig, y_raw, y_ranked)
    end

    opt_options = Optim.Options(iterations = 200,
                                show_trace  = false,
                                x_abstol    = 1e-5,
                                f_reltol    = 1e-6)

    try
        result = Optim.optimize(loss, consts0, NelderMead(), opt_options)
        inject_constants!(tree, Optim.minimizer(result))
        return true
    catch
        # Preserve original constants on any error (NaN input, degenerate tree…)
        return false
    end
end


# =============================================================================
# 3. Individual-level and population-level entry points
# =============================================================================

"""
    optimize_individual_constants!(ind, X, y_raw, y_ranked, config) → Nothing

Apply `optimize_constants!` to every tree in `ind`.
`y_ranked` should be `prerank_y(y_raw)`, pre-computed by the caller.
"""
function optimize_individual_constants!(ind::Individual,
                                         X::Matrix{Float64},
                                         y_raw::Vector{Float64},
                                         y_ranked::Vector{Float64},
                                         config::GPConfig)
    for tree in ind.trees
        optimize_constants!(tree, X, y_raw, y_ranked, config)
    end
    nothing
end


"""
    optimize_population_constants!(population, X, y_raw, y_ranked, config, gen) → Nothing

Apply constant optimization to the top `elite_count` individuals.
Only runs on generations where `gen % const_opt_every_n_gens == 0`.
Uses `Threads.@threads` — each individual is fully independent.

`y_ranked` should be the same ranked vector used by `evaluate_population!`
on the same subsampled `X` and `y_raw`, so PPS is evaluated on a consistent
data slice.
"""
function optimize_population_constants!(population::Vector{Individual},
                                         X::Matrix{Float64},
                                         y_raw::Vector{Float64},
                                         y_ranked::Vector{Float64},
                                         config::GPConfig,
                                         gen::Int)
    gen % config.const_opt_every_n_gens == 0 || return nothing

    order   = sortperm(population; by = ind -> ind.fitness, rev = true)
    n_opt   = min(config.elite_count, length(population))
    targets = order[1:n_opt]

    Threads.@threads for i in targets
        optimize_individual_constants!(population[i], X, y_raw, y_ranked, config)
    end

    nothing
end