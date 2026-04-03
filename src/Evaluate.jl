# =============================================================================
#Evaluate.jl — constant I/O, multi-tree combination, behavior fingerprinting
#
# Public API
# ─────────────────────────────────────────────────────────────────────────────
#   extract_constants(tree)               → Vector{Float64}
#   inject_constants!(tree, consts)       → Nothing
#   update_complexity!(ind)               → Nothing
#   combine_trees(ind, X)                 → Vector{Float64}
#   update_behavior!(ind, X, beh_rows)    → Nothing
# =============================================================================


# =============================================================================
# 1. Constant extraction / injection  (used by Constants.jl / Optim.jl)
# ========================

"""
    extract_constants(tree) -> Vector{Float64}

Returns a copy of all numeric constants in 'tree.constants'.
The returned vector is passed directlly to the optimizer - mutating it does not affect the tree until 'inject_constants!' is called.
"""
extract_constants(tree::Tree)::Vector{Float64} = copy(tree.constants)


"""
    inject_constants!(tree, consts)

Overwrites tree.constants with the values in 'consts'. 
Called after optimizer returns the optimized parameter vector.
"""
function inject_constants!(tree::Tree, consts::Vector{Float64})
    length(consts) == length(tree.constants) || error("inject_constants!: length mismatch - " * "tree has $(length(tree.constants)) constants, but got $(length(consts))")
    copyto!(tree.constants, consts)
    nothing
end


# =============================================================================
# 2. Complexity tracking
# =============================================================================

"""
    update_complexity!(ind)
Recomputes the complexity of 'ind' as the total number of nodes across all trees and updates the 'complexity' field.
"""
function update_complexity!(ind::Individual)
    ind.complexity = sum(t.size for t in ind.trees; init = 0)
    nothing
end






# =============================================================================
# 3. Multi-tree combination
# =============================================================================

"""
    combine_trees(ind, X) -> Vector{Float64}

Evaluate every tree in `ind` on `X` and combine their outputs according to
`ind.combination`:

  :weighted_sum  →  Σᵢ wᵢ · treeᵢ(X)
                    Weights co-optimised with constants.

  :rank_average  →  mean( cs_rank(treeᵢ(X)) )
                    Each tree contributes its percentile-rank signal.
                    Robust to outliers; no weight tuning required.

  :vote          →  sign( Σᵢ sign(treeᵢ(X)) )
                    Directional majority vote; interpretable.

Edge case: empty `ind.trees` returns a zero vector of length `size(X, 1)`.
"""    

function combine_trees(ind::Individual, X::Matrix{Float64})::Vector{Float64}
    isempty(ind.trees) && return zeros(Float64, size(X, 1))
    n = size(X, 1)

    if ind.combination == :weighted_sum
        signal = zeros(Float64, n)
        for (tree, w) in zip(ind.trees, ind.weights)
            signal .+= w .* eval_tree(tree, X)
        end
        return signal

    elseif ind.combination == :rank_average
        signal = zeros(Float64, n)
        for tree in ind.trees
            signal .+= cs_rank(eval_tree(tree, X))
        end
        signal ./= length(ind.trees)
        return signal
        
        
    else  # :vote
        votes = zeros(Float64, n)
        for tree in ind.trees
            votes .+= sign.(eval_tree(tree, X))
        end
        return sign.(votes)
    end
end




# =============================================================================
# 4. Behavior fingerprint (for diversity calculation)
# =============================================================================
"""
    update_behavior!(ind, X, beh_rows)

Evaluate 'ind' on the fixed 'beh_rows' of subset of "X" and store the compressed output in 'ind.behavior'.

"""

function update_behavior!(ind::Individual, X::Matrix{Float64},
                          beh_rows::Vector{Int})
    X_beh       = X[beh_rows, :]
    signal      = combine_trees(ind, X_beh)
    ind.behavior = Float32.(signal)
    nothing
end