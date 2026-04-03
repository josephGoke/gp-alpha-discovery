# =============================================================================
# Operators_GP.jl — Genetic operators: crossover + 4-operator mutation suite
#
# Public API
# ─────────────────────────────────────────────────────────────────────────────
#   crossover(rng, p1, p2, config)                    → (Individual, Individual)
#   subtree_mutate(rng, ind, config, n_feat, op_sets) → Individual
#   point_mutate(rng, ind, config, n_feat, op_sets)   → Individual
#   hoist_mutate(rng, ind, config)                    → Individual
#   shrink_mutate(rng, ind, config, n_feat, op_sets)  → Individual
#   apply_mutation(rng, ind, config, n_feat, op_sets) → Individual
#
# Convention: all mutation functions return a new Individual (they do NOT
# mutate in-place). The original `ind` is never modified.
#
# Key invariants upheld
# ─────────────────────────────────────────────────────────────────────────────
#   1. No child tree ever exceeds config.max_depth (depth-bounded crossover).
#   2. Each mutation operator touches exactly one randomly-selected node;
#      it does NOT recurse down the whole tree (V1's catastrophic mistake).
#   3. All random state goes through the rng argument — no bare rand() calls.
# =============================================================================

using Random


# =============================================================================
# Module-level tunables
# =============================================================================

const _CROSSOVER_MAX_RETRIES = 3      # retries before falling back to clone
const _POINT_MUTATE_SIGMA    = 0.1    # stddev for constant-nudging in point_mutate!


# =============================================================================
# 1. Private helpers
# =============================================================================

"""
    _clone_individual(ind) → Individual

Return a fully independent deep copy of `ind`.
`copy_tree` is O(n) per tree; `Node` is `isbits` so `copy(nodes)` is a
value copy with no heap sharing.
"""
function _clone_individual(ind::Individual)::Individual
    Individual(
        copy_tree.(ind.trees),
        copy(ind.weights),
        ind.combination,
        ind.fitness,
        ind.complexity,
        ind.age,
        copy(ind.behavior),
        ind.dominated,
        ind.pps_score,
        ind.rre_score
    )
end


"""
    _extract_subtree(tree, root_idx) → Tree

Build an independent `Tree` from the subtree of `tree` rooted at `root_idx`.

Index remapping
───────────────
`subtree_indices` returns nodes in DFS pre-order (parent before children),
so position 1 in the new node array = root, and all child references
(which always point to later-visited nodes) map to higher positions.

TS-node const_idx (window d) is preserved unchanged.
Only NODE_CONSTANT const_idx values are remapped into the new constants vector.
"""
function _extract_subtree(tree::Tree, root_idx::Int)::Tree
    indices    = subtree_indices(tree.nodes, root_idx)   # DFS pre-order
    n          = length(indices)
    old_to_new = Dict{Int,Int}(old => new for (new, old) in enumerate(indices))

    old_ci_to_new = Dict{Int,Int}()
    new_constants  = Float64[]
    new_nodes      = Vector{Node}(undef, n)

    for (new_pos, old_idx) in enumerate(indices)
        nd  = tree.nodes[old_idx]
        nl  = nd.left  == Int16(0) ? Int16(0) : Int16(old_to_new[Int(nd.left)])
        nr  = nd.right == Int16(0) ? Int16(0) : Int16(old_to_new[Int(nd.right)])

        # Remap const_idx only for leaf-constant nodes; TS window d is NOT
        # in the constants vector — it's a raw integer in the same field.
        nci = nd.const_idx
        if nd.node_type == NODE_CONSTANT
            ci = Int(nd.const_idx)
            if !haskey(old_ci_to_new, ci)
                push!(new_constants, tree.constants[ci])
                old_ci_to_new[ci] = length(new_constants)
            end
            nci = Int16(old_ci_to_new[ci])
        end

        new_nodes[new_pos] = Node(nd.node_type, nd.op_idx, nl, nr, nd.feature, nci)
    end

    t = Tree(new_nodes, new_constants, n, 0)
    recompute_depth!(t)
    return t
end


"""
    _node_depth(tree, target_idx) → Int

Return the depth of node `target_idx` within `tree` (root = 0).
Iterative DFS with an explicit (node_idx, depth) stack — O(n) worst case.
"""
function _node_depth(tree::Tree, target_idx::Int)::Int
    stack = Tuple{Int,Int}[(1, 0)]
    while !isempty(stack)
        idx, d = pop!(stack)
        idx == target_idx && return d
        nd = tree.nodes[idx]
        is_operator(nd) || continue
        nd.right != Int16(0) && push!(stack, (Int(nd.right), d + 1))
        nd.left  != Int16(0) && push!(stack, (Int(nd.left),  d + 1))
    end
    error("_node_depth: target_idx=$target_idx not found in tree of size $(tree.size)")
end


# =============================================================================
# 2. Crossover — depth-bounded subtree crossover
# =============================================================================

"""
    crossover(rng, parent1, parent2, config) → (Individual, Individual)

Multi-tree depth-bounded subtree crossover.

One tree (randomly selected index) is crossed; the remaining trees are
inherited intact from their respective parents.

Depth enforcement
─────────────────
Retry up to `_CROSSOVER_MAX_RETRIES` times if the resulting child tree would
exceed `config.max_depth`. On exhaustion, both children keep their original
cloned trees at that index (no invalid offspring ever enters the population).

Depth constraint: insertion_depth + donor_depth ≤ max_depth
"""
function crossover(rng::AbstractRNG, parent1::Individual, parent2::Individual,
                   config::GPConfig)::Tuple{Individual, Individual}
    child1 = _clone_individual(parent1)
    child2 = _clone_individual(parent2)

    tree_idx = rand(rng, 1:length(parent1.trees))
    t1       = child1.trees[tree_idx]
    t2       = child2.trees[tree_idx]

    new_t1 = nothing
    new_t2 = nothing

    for _ in 1:_CROSSOVER_MAX_RETRIES
        idx1   = random_node_idx(rng, t1)
        idx2   = random_node_idx(rng, t2)
        d1     = _node_depth(t1, idx1)
        d2     = _node_depth(t2, idx2)
        donor1 = _extract_subtree(t2, idx2)   # subtree from t2 → insert into t1 at idx1
        donor2 = _extract_subtree(t1, idx1)   # subtree from t1 → insert into t2 at idx2

        if d1 + donor1.depth <= config.max_depth &&
           d2 + donor2.depth <= config.max_depth
            new_t1 = replace_subtree(t1, idx1, donor1)
            new_t2 = replace_subtree(t2, idx2, donor2)
            break
        end
    end

    if !isnothing(new_t1)
        child1.trees[tree_idx] = new_t1
        child2.trees[tree_idx] = new_t2
    end

    child1.complexity = sum(t.size for t in child1.trees; init=0)
    child2.complexity = sum(t.size for t in child2.trees; init=0)
    return child1, child2
end


# =============================================================================
# 3. Mutation 1: subtree_mutate
# =============================================================================

"""
    subtree_mutate(rng, ind, config, n_features, op_sets) → Individual

Pick a random node; replace its entire subtree with a freshly-generated
random tree whose depth is bounded so the result never exceeds `max_depth`.

    replacement_max_depth = config.max_depth - depth(target_node)

Minimum allowed replacement depth is 0 (a single leaf is valid).
Uses `:grow` method for the replacement so size stays diverse.
"""
function subtree_mutate(rng::AbstractRNG, ind::Individual, config::GPConfig,
                         n_features::Int, op_sets::NamedTuple)::Individual
    child    = _clone_individual(ind)
    tree_idx = rand(rng, 1:length(child.trees))
    tree     = child.trees[tree_idx]

    target = random_node_idx(rng, tree)
    avail  = max(config.max_depth - _node_depth(tree, target), 0)

    donor = build_random_tree(rng, n_features, config, op_sets;
                              method = :grow, max_depth = avail)

    child.trees[tree_idx] = replace_subtree(tree, target, donor)
    child.complexity = sum(t.size for t in child.trees; init=0)
    return child
end


# =============================================================================
# 4. Mutation 2: point_mutate
# =============================================================================

"""
    point_mutate(rng, ind, config, n_features, op_sets) → Individual

Fine-grained single-node mutation. Picks a random node:

- **Operator node**: swap to a different operator in the same tier.
  Children and (for TS nodes) the window parameter d are unchanged.
- **Constant node**: nudge by `randn(rng) * _POINT_MUTATE_SIGMA`.
- **Variable node**: swap to a different feature (no-op if only 1 feature).

Tier grouping:
    elementwise unary  ↔  op_sets.unary
    elementwise binary ↔  op_sets.binary
    ts_unary           ↔  op_sets.ts_unary   (window d preserved)
    ts_binary          ↔  op_sets.ts_binary  (window d preserved)
    cross-sectional    ↔  op_sets.cs
"""
function point_mutate(rng::AbstractRNG, ind::Individual, config::GPConfig,
                       n_features::Int, op_sets::NamedTuple)::Individual
    child    = _clone_individual(ind)
    tree_idx = rand(rng, 1:length(child.trees))
    tree     = child.trees[tree_idx]

    target = random_node_idx(rng, tree)
    nd     = tree.nodes[target]

    if is_operator(nd)
        pool = if is_ts_unary_op(nd.op_idx)
            op_sets.ts_unary
        elseif is_ts_binary_op(nd.op_idx)
            op_sets.ts_binary
        elseif is_cs_op(nd.op_idx) || is_unary_op(nd.op_idx)
            vcat(op_sets.unary, op_sets.cs)
        else
            op_sets.binary
        end

        length(pool) <= 1 && return child   # nothing to swap

        new_op = nd.op_idx
        while new_op == nd.op_idx
            new_op = rand(rng, pool)
        end

        tree.nodes[target] = Node(nd.node_type, new_op, nd.left, nd.right,
                                  nd.feature, nd.const_idx)

    elseif is_constant(nd)
        tree.constants[Int(nd.const_idx)] += randn(rng) * _POINT_MUTATE_SIGMA

    else  # variable node
        n_features == 1 && return child
        new_feat = Int(nd.feature)
        while new_feat == Int(nd.feature)
            new_feat = rand(rng, 1:n_features)
        end
        tree.nodes[target] = Node(nd.node_type, nd.op_idx, nd.left, nd.right,
                                  Int16(new_feat), nd.const_idx)
    end

    return child
end


# =============================================================================
# 5. Mutation 3: hoist_mutate
# =============================================================================

"""
    hoist_mutate(rng, ind, config) → Individual

Pick a random non-terminal node; replace that node's subtree with one of its
own child subtrees, effectively hoisting a sub-expression upward.

Effect: tree gets smaller → bloat control. The computation is simplified
by discarding the outer (more complex) wrapping expression.
Falls back to a clone if no operator node exists (single-leaf tree).
"""
function hoist_mutate(rng::AbstractRNG, ind::Individual,
                       config::GPConfig)::Individual
    child    = _clone_individual(ind)
    tree_idx = rand(rng, 1:length(child.trees))
    tree     = child.trees[tree_idx]

    target = random_nonterminal_idx(rng, tree)
    isnothing(target) && return child

    nd = tree.nodes[target]

    # Collect non-null child root indices
    candidates = Int[]
    nd.left  != Int16(0) && push!(candidates, Int(nd.left))
    nd.right != Int16(0) && push!(candidates, Int(nd.right))
    isempty(candidates) && return child   # safety — should never trigger

    hoist_root = rand(rng, candidates)
    donor = _extract_subtree(tree, hoist_root)

    child.trees[tree_idx] = replace_subtree(tree, target, donor)
    child.complexity = sum(t.size for t in child.trees; init=0)
    return child
end


# =============================================================================
# 6. Mutation 4: shrink_mutate
# =============================================================================

"""
    shrink_mutate(rng, ind, config, n_features, op_sets) → Individual

Pick a random non-terminal node; replace its entire subtree with a single
random leaf (variable or constant, drawn via `build_random_tree` at depth 0).

More aggressive than `hoist_mutate` — discards the child structure entirely.
Useful for removing deeply nested redundant subexpressions.
Falls back to a clone if no operator node exists.
"""
function shrink_mutate(rng::AbstractRNG, ind::Individual, config::GPConfig,
                        n_features::Int, op_sets::NamedTuple)::Individual
    child    = _clone_individual(ind)
    tree_idx = rand(rng, 1:length(child.trees))
    tree     = child.trees[tree_idx]

    target = random_nonterminal_idx(rng, tree)
    isnothing(target) && return child

    leaf = build_random_tree(rng, n_features, config, op_sets; max_depth = 0)
    child.trees[tree_idx] = replace_subtree(tree, target, leaf)
    child.complexity = sum(t.size for t in child.trees; init=0)
    return child
end


# =============================================================================
# 7. Dispatcher: apply_mutation
# =============================================================================

"""
    apply_mutation(rng, ind, config, n_features, op_sets) → Individual

Sample one mutation operator from `config.mutation_prob` and apply it.

Cumulative probability thresholds:
    [0,          subtree)                       → subtree_mutate!
    [subtree,    subtree+point)                 → point_mutate!
    [+point,     subtree+point+hoist)           → hoist_mutate!
    [+hoist,     subtree+point+hoist+shrink)    → shrink_mutate!
    [total, 1.0]                                → clone (no mutation)
"""
function apply_mutation(rng::AbstractRNG, ind::Individual, config::GPConfig,
                        n_features::Int, op_sets::NamedTuple)::Individual
    mp = config.mutation_prob
    r  = rand(rng)

    t1 = mp.subtree
    t2 = t1 + mp.point
    t3 = t2 + mp.hoist
    t4 = t3 + mp.shrink

    r < t1 && return subtree_mutate(rng, ind, config, n_features, op_sets)
    r < t2 && return point_mutate(rng, ind, config, n_features, op_sets)
    r < t3 && return hoist_mutate(rng, ind, config)
    r < t4 && return shrink_mutate(rng, ind, config, n_features, op_sets)
    return _clone_individual(ind)
end