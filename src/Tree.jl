# =============================================================================
# Tree.jl — Flat-array GP tree: construction, evaluation, manipulation
#
# Layout invariant
# ─────────────────────────────────────────────────────────────────────────────
#   nodes[1]  is always the root.
#   Children are referenced by their 1-based index into nodes[].
#   Index 0 (Int16) is the null sentinel — no child.
#   Trees are built in insertion order (pre-order):
#     root placeholder pushed first, then left subtree, then right subtree.
#     After children are built their starting indices are known, so the
#     placeholder is overwritten with the correct op_node/unary_node.
#
# 
# ─────────────────────────────────────────────────────────────────────────────
#   make_op_sets(config)                              → NamedTuple
#   build_random_tree(rng, n_features, config, op_sets
#                     [; method, max_depth])          → Tree
#   eval_tree(tree, X)                                → Vector{Float64}
#   copy_tree(tree)                                   → Tree
#   subtree_indices(nodes, root_idx)                  → Vector{Int}
#   subtree_size(nodes, root_idx)                     → Int
#   compute_depth(nodes, idx)                         → Int
#   recompute_depth!(tree)                            → Nothing
#   replace_subtree(tree, target_idx, donor)          → Tree
#   random_node_idx(rng, tree)                        → Int
#   random_nonterminal_idx(rng, tree)                 → Union{Int,Nothing}
#   tree_to_string(tree[, idx])                       → String
# =============================================================================

using Random


# =============================================================================
# 1. Operator-set resolver (convenience wrapper around validate_op_symbols)
# =============================================================================

"""
    make_op_sets(config) → NamedTuple

Resolve all operator Symbol names in `config` to Int8 index vectors.
Returns `(unary, binary, ts_unary, ts_binary, cs)`.
Call once at engine startup; pass the result into every tree-building call.
"""
function make_op_sets(config::GPConfig)::NamedTuple
    u, b, tu, tb, cs = validate_op_symbols(config)
    return (unary=u, binary=b, ts_unary=tu, ts_binary=tb, cs=cs)
end


# =============================================================================
# 2. Subtree utilities
# =============================================================================

"""
    subtree_indices(nodes, root_idx) → Vector{Int}

Return all node indices in the subtree rooted at `root_idx` (DFS, root first).
"""
function subtree_indices(nodes::Vector{Node}, root_idx::Int)::Vector{Int}
    result = Int[]
    stack  = Int[root_idx]
    while !isempty(stack)
        idx  = pop!(stack)
        push!(result, idx)
        n = nodes[idx]
        is_operator(n) || continue
        n.right != Int16(0) && push!(stack, Int(n.right))
        n.left  != Int16(0) && push!(stack, Int(n.left))
    end
    return result
end

"""
    subtree_size(nodes, root_idx) → Int

Count nodes in the subtree rooted at `root_idx`.
"""
function subtree_size(nodes::Vector{Node}, root_idx::Int)::Int
    root_idx == 0 && return 0
    n = nodes[root_idx]
    is_leaf(n) && return 1
    sz = 1
    n.left  != Int16(0) && (sz += subtree_size(nodes, Int(n.left)))
    n.right != Int16(0) && (sz += subtree_size(nodes, Int(n.right)))
    return sz
end

"""
    compute_depth(nodes, idx) → Int

Recursively compute depth of subtree rooted at `idx`.
Leaves = 0. Null index (0) = -1 (used by parent to get max correctly).
"""
function compute_depth(nodes::Vector{Node}, idx::Int)::Int
    idx == 0 && return -1
    n = nodes[idx]
    is_leaf(n) && return 0
    ld = n.left  != Int16(0) ? compute_depth(nodes, Int(n.left))  : -1
    rd = n.right != Int16(0) ? compute_depth(nodes, Int(n.right)) : -1
    return 1 + max(ld, rd)
end

"""
    recompute_depth!(tree)

Recompute and update `tree.size` and `tree.depth` in-place.
Must be called after any structural mutation.
"""
function recompute_depth!(tree::Tree)
    tree.size  = subtree_size(tree.nodes, 1)
    tree.depth = compute_depth(tree.nodes, 1)
    nothing
end


# =============================================================================
# 3. Random node sampling helpers (used by mutation & crossover)
# =============================================================================

"""
    random_node_idx(rng, tree) → Int

Uniformly sample a node index from `tree`.
"""
random_node_idx(rng::AbstractRNG, tree::Tree)::Int = rand(rng, 1:tree.size)

"""
    random_nonterminal_idx(rng, tree) → Union{Int, Nothing}

Uniformly sample a non-leaf (operator) node index.
Returns `nothing` if the tree is a single leaf (no operators).
"""
function random_nonterminal_idx(rng::AbstractRNG, tree::Tree)::Union{Int, Nothing}
    ops = [i for i in 1:tree.size if is_operator(tree.nodes[i])]
    isempty(ops) ? nothing : rand(rng, ops)
end


# =============================================================================
# 4. Tree construction — internals
# =============================================================================

@inline function _push_leaf!(nodes::Vector{Node}, constants::Vector{Float64},
                              rng::AbstractRNG, n_features::Int, config::GPConfig)
    if rand(rng) < config.const_prob
        push!(constants, randn(rng))
        push!(nodes, constant_node(length(constants)))
    else
        push!(nodes, variable_node(rand(rng, 1:n_features)))
    end
end

"""
    _build_node!(nodes, constants, rng, n_features, config, op_sets, depth, max_depth, method)

Recursive pre-order tree builder. Pushes nodes into `nodes` and constants into
`constants`. Uses a "reserve-then-overwrite" approach so child indices are
known before the parent node is finalised.
"""
function _build_node!(nodes::Vector{Node}, constants::Vector{Float64},
                      rng::AbstractRNG, n_features::Int, config::GPConfig,
                      op_sets::NamedTuple, depth::Int, max_depth::Int, method::Symbol)

    must_leaf = (depth >= max_depth)
    must_op   = (depth < config.min_depth)

    # Decide leaf vs operator ─────────────────────────────────────────────────
    # :full  → always operator until max_depth, then always leaf
    # :grow  → 50 / 50 between min_depth and max_depth
    make_leaf = must_leaf ||
                (!must_op && method == :grow && rand(rng) < 0.5)

    if make_leaf
        _push_leaf!(nodes, constants, rng, n_features, config)
        return
    end

    # Reserve slot for this operator node ─────────────────────────────────────
    my_idx = length(nodes) + 1
    push!(nodes, Node(NODE_OPERATOR, Int8(0), Int16(0), Int16(0), Int16(0), Int16(0)))

    # Pick operator ───────────────────────────────────────────────────────────
    has_fin  = !isempty(op_sets.ts_unary) || !isempty(op_sets.ts_binary) || !isempty(op_sets.cs)
    use_fin  = has_fin && rand(rng) < config.fin_op_prob

    op_idx::Int8 = if use_fin
        rand(rng, vcat(op_sets.ts_unary, op_sets.ts_binary, op_sets.cs))
    else
        rand(rng, vcat(op_sets.unary, op_sets.binary))
    end

    # Build children, then overwrite placeholder ───────────────────────────────
    if is_ts_unary_op(op_idx)
        # Child is a full recursive subtree — enables e.g ts_mean(safe_add(x1,x2), d)
        var_idx = Int16(length(nodes) + 1)
        _build_node!(nodes, constants, rng, n_features, config, op_sets, depth + 1, max_depth, method)

        d = rand(rng, config.ts_window_sizes)
        nodes[my_idx] = Node(NODE_OPERATOR, op_idx, var_idx, Int16(0), Int16(0), Int16(d))

    elseif is_ts_binary_op(op_idx)
        var1_idx = Int16(length(nodes) + 1)
        _build_node!(nodes, constants, rng, n_features, config, op_sets, depth + 1, max_depth, method)
        var2_idx = Int16(length(nodes) + 1)
        _build_node!(nodes, constants, rng, n_features, config, op_sets, depth + 1, max_depth, method)

        d = rand(rng, config.ts_window_sizes)
        nodes[my_idx] = Node(NODE_OPERATOR, op_idx, var1_idx, var2_idx, Int16(0), Int16(d))

    elseif is_cs_op(op_idx)
        left_idx = Int16(length(nodes) + 1)
        _build_node!(nodes, constants, rng, n_features, config, op_sets, depth + 1, max_depth, method)
        nodes[my_idx] = Node(NODE_OPERATOR, op_idx, left_idx, Int16(0), Int16(0), Int16(0))

    elseif is_unary_op(op_idx)
        left_idx = Int16(length(nodes) + 1)
        _build_node!(nodes, constants, rng, n_features, config, op_sets, depth + 1, max_depth, method)
        nodes[my_idx] = Node(NODE_OPERATOR, op_idx, left_idx, Int16(0), Int16(0), Int16(0))

    else  # binary elementwise
        left_idx = Int16(length(nodes) + 1)
        _build_node!(nodes, constants, rng, n_features, config, op_sets, depth + 1, max_depth, method)
        right_idx = Int16(length(nodes) + 1)
        _build_node!(nodes, constants, rng, n_features, config, op_sets, depth + 1, max_depth, method)
        nodes[my_idx] = Node(NODE_OPERATOR, op_idx, left_idx, right_idx, Int16(0), Int16(0))
    end
end


# =============================================================================
# 5. Tree construction — entry point
# =============================================================================

"""
    build_random_tree(rng, n_features, config, op_sets;
                      method=:grow, max_depth=config.max_depth) → Tree

Build a random expression tree.

Arguments
---------
- `rng`        : seeded `AbstractRNG` for reproducibility
- `n_features` : number of input features (columns in X)
- `config`     : `GPConfig` — reads `min_depth`, `const_prob`,              `fin_op_prob`, `ts_window_sizes`
- `op_sets`    : NamedTuple from `make_op_sets(config)` — resolved Int8 indices
- `method`     : `:grow` (probabilistic) or `:full` (all leaves at max_depth)
- `max_depth`  : overrides `config.max_depth` when provided
"""
function build_random_tree(rng::AbstractRNG, n_features::Int, config::GPConfig,
                           op_sets::NamedTuple;
                           method::Symbol = :grow,
                           max_depth::Int = config.max_depth)::Tree
    method in (:grow, :full) ||
        error("build_random_tree: method must be :grow or :full (got :$method)")
    n_features >= 1 ||
        error("build_random_tree: n_features must be >= 1 (got $n_features)")

    nodes     = Node[]
    constants = Float64[]
    sizehint!(nodes, 2^(max_depth + 1))   # avoids most reallocations

    _build_node!(nodes, constants, rng, n_features, config, op_sets, 0, max_depth, method)

    sz = length(nodes)
    dp = compute_depth(nodes, 1)
    return Tree(nodes, constants, sz, dp)
end


# =============================================================================
# 6. Tree evaluation
# =============================================================================

"""
    eval_node(tree, X, idx) → Vector{Float64}

Recursively evaluate the subtree rooted at `idx`.
Returns a signal vector of length `size(X, 1)`.
"""
function eval_node(tree::Tree, X::Matrix{Float64}, idx::Int)::Vector{Float64}
    n = tree.nodes[idx]

    # ── Leaves ────────────────────────────────────────────────────────────────
    if is_variable(n)
        return X[:, Int(n.feature)]

    elseif is_constant(n)
        return fill(tree.constants[Int(n.const_idx)], size(X, 1))

    # ── Tier-1 elementwise operators ──────────────────────────────────────────
    elseif is_unary_op(n.op_idx)
        child = eval_node(tree, X, Int(n.left))
        return apply_unary_op(n.op_idx, child)

    elseif is_binary_op(n.op_idx)
        lv = eval_node(tree, X, Int(n.left))
        rv = eval_node(tree, X, Int(n.right))
        return apply_binary_op(n.op_idx, lv, rv)

    # ── Tier-2 time-series operators ──────────────────────────────────────────
    elseif is_ts_unary_op(n.op_idx)
        # col = Int(tree.nodes[Int(n.left)].feature)
        child = eval_node(tree, X, Int(n.left))
        d   = Int(n.const_idx)
        return apply_ts_unary_op(n.op_idx, child, d)

    elseif is_ts_binary_op(n.op_idx)
        child1 = eval_node(tree, X, Int(n.left))
        child2 = eval_node(tree, X, Int(n.right))
        d    = Int(n.const_idx)
        return apply_ts_binary_op(n.op_idx, child1, child2, d)

    # ── Tier-2 cross-sectional operators ─────────────────────────────────────
    elseif is_cs_op(n.op_idx)
        child = eval_node(tree, X, Int(n.left))
        return apply_cs_op(n.op_idx, child)
    end

    error("eval_node: unhandled node at idx=$idx, op_idx=$(n.op_idx)")
end

"""
    eval_tree(tree, X) → Vector{Float64}

Evaluate `tree` on data matrix `X` (rows = samples, cols = features).
"""
eval_tree(tree::Tree, X::Matrix{Float64})::Vector{Float64} = eval_node(tree, X, 1)


# =============================================================================
# 7. Tree copying
# =============================================================================

"""
    copy_tree(tree) → Tree

Return a fully independent copy of `tree`.
Safe because `Node` is `isbits` — `copy(tree.nodes)` is a true value copy.
"""
copy_tree(tree::Tree)::Tree =
    Tree(copy(tree.nodes), copy(tree.constants), tree.size, tree.depth)


# =============================================================================
# 8. Subtree replacement (foundation for crossover and mutation)
# =============================================================================

"""
    replace_subtree(tree, target_idx, donor) → Tree

Return a new Tree in which the subtree rooted at `target_idx` is replaced by
the entirety of `donor`.

Algorithm (O(n))
----------------
1. Identify removed nodes via DFS on the old subtree.
2. Build `old_to_new[i]`: new index of each kept node, plus
   `old_to_new[target_idx] = n_kept + 1` (donor root position).
3. Rebuild node array as [remapped kept nodes ++ offset donor nodes].
4. Merge constant arrays; only NODE_CONSTANT const_idx values are offset —
   TS-node const_idx values (window d) are NOT touched.
5. Recompute size and depth.

Note: constants from the removed subtree are retained in the array but
become unreachable. This is acceptable for typical tree sizes; a compaction
pass can be added later if memory pressure is observed.
"""
function replace_subtree(tree::Tree, target_idx::Int, donor::Tree)::Tree
    1 <= target_idx <= tree.size ||
        throw(BoundsError("replace_subtree: target_idx=$target_idx out of [1,$(tree.size)]"))

    # Step 1 — identify removed nodes ─────────────────────────────────────────
    removed = Set(subtree_indices(tree.nodes, target_idx))
    n_kept  = tree.size - length(removed)
    n_donor = donor.size

    # Step 2 — build old→new index mapping ────────────────────────────────────
    old_to_new = zeros(Int16, tree.size)
    k = 0
    for i in 1:tree.size
        i ∈ removed && continue
        k += 1
        old_to_new[i] = Int16(k)
    end
    # target_idx is in removed; remap it to donor root position
    old_to_new[target_idx] = Int16(n_kept + 1)

    # Step 3 — build result node array ────────────────────────────────────────
    result = Vector{Node}(undef, n_kept + n_donor)

    # kept nodes: remap their left/right references
    k = 0
    for i in 1:tree.size
        i ∈ removed && continue
        k += 1
        n  = tree.nodes[i]
        nl = n.left  == Int16(0) ? Int16(0) : old_to_new[Int(n.left)]
        nr = n.right == Int16(0) ? Int16(0) : old_to_new[Int(n.right)]
        result[k] = Node(n.node_type, n.op_idx, nl, nr, n.feature, n.const_idx)
    end

    # donor nodes: offset left/right by n_kept; offset const_idx only for
    # NODE_CONSTANT nodes (TS nodes store window d there — do NOT offset)
    const_offset = Int16(length(tree.constants))
    for j in 1:n_donor
        n  = donor.nodes[j]
        nl = n.left  == Int16(0) ? Int16(0) : Int16(Int(n.left)  + n_kept)
        nr = n.right == Int16(0) ? Int16(0) : Int16(Int(n.right) + n_kept)
        nc = (n.node_type == NODE_CONSTANT) ?
             Int16(Int(n.const_idx) + const_offset) : n.const_idx
        result[n_kept + j] = Node(n.node_type, n.op_idx, nl, nr, n.feature, nc)
    end

    # Step 4 — merge constant arrays ──────────────────────────────────────────
    new_constants = vcat(tree.constants, donor.constants)

    # Step 5 — assemble and return ────────────────────────────────────────────
    new_tree = Tree(result, new_constants, n_kept + n_donor, 0)
    recompute_depth!(new_tree)
    return new_tree
end


# =============================================================================
# 9. Pretty-printing
# =============================================================================

"""
    tree_to_string(tree[, idx=1]) → String

Recursive infix/functional string for debugging and HoF reporting.
"""
function tree_to_string(tree::Tree, idx::Int = 1)::String
    idx == 0 && return ""
    n = tree.nodes[idx]

    if is_variable(n)
        return "x$(Int(n.feature))"

    elseif is_constant(n)
        return string(round(tree.constants[Int(n.const_idx)]; digits = 4))

    else
        name = op_name(n.op_idx)

        if is_ts_unary_op(n.op_idx)
            inner = tree_to_string(tree, Int(n.left))
            return "$name($inner, d=$(Int(n.const_idx)))"

        elseif is_ts_binary_op(n.op_idx)
            l = tree_to_string(tree, Int(n.left))
            r = tree_to_string(tree, Int(n.right))
            return "$name($l, $r, d=$(Int(n.const_idx)))"

        elseif is_unary_op(n.op_idx) || is_cs_op(n.op_idx)
            return "$name($(tree_to_string(tree, Int(n.left))))"

        else  # binary elementwise
            l = tree_to_string(tree, Int(n.left))
            r = tree_to_string(tree, Int(n.right))
            return "$name($l, $r)"
        end
    end
end

function Base.show(io::IO, tree::Tree)
    print(io, "Tree(size=$(tree.size), depth=$(tree.depth)): $(tree_to_string(tree))")
end