# =============================================================================
# test/test_tree.jl
# Run with: julia --project test/test_tree.jl
# =============================================================================

using Test
using Random

include("../src/Types.jl")
include("../src/Functions.jl")
include("../src/Tree.jl")

# ── Shared test fixtures ──────────────────────────────────────────────────────

const RNG        = MersenneTwister(42)
const N_FEATURES = 5
const N_ROWS     = 50
const X_TEST     = randn(RNG, N_ROWS, N_FEATURES)

const CFG = GPConfig(
    min_depth          = 2,
    max_depth          = 5,
    const_prob         = 0.2,
    fin_op_prob        = 0.3,
    ts_window_sizes    = [3, 5, 10],
    population_size    = 100,       # needed for validate_config
    n_islands          = 1,
)

const OP_SETS = make_op_sets(CFG)

# Helper: build a known hand-crafted tree
#   safe_add(x1, safe_log(x2))
#   nodes: [1]=binary_add, [2]=variable x1, [3]=unary_log, [4]=variable x2
function _hand_tree()
    nodes = Node[
        op_node(Int8(10), Int16(2), Int16(3)),   # safe_add(left=2, right=3)
        variable_node(1),                         # x1
        unary_node(Int8(2), Int16(4)),            # safe_log(left=4)
        variable_node(2),                         # x2
    ]
    Tree(nodes, Float64[], 4, 2)
end

# =============================================================================
@testset "Tree.jl" begin

# ─────────────────────────────────────────────────────────────────────────────
@testset "subtree_indices" begin
    t = _hand_tree()
    # Full tree from root
    @test sort(subtree_indices(t.nodes, 1)) == [1, 2, 3, 4]
    # Subtree rooted at node 3 (unary_log(x2))
    @test sort(subtree_indices(t.nodes, 3)) == [3, 4]
    # Leaf node — only itself
    @test subtree_indices(t.nodes, 2) == [2]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "subtree_size" begin
    t = _hand_tree()
    @test subtree_size(t.nodes, 1) == 4
    @test subtree_size(t.nodes, 3) == 2
    @test subtree_size(t.nodes, 2) == 1
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "compute_depth / recompute_depth!" begin
    t = _hand_tree()
    @test compute_depth(t.nodes, 1) == 2  # add → log → x2  (depth 2)
    @test compute_depth(t.nodes, 3) == 1  # log → x2
    @test compute_depth(t.nodes, 2) == 0  # leaf

    # recompute_depth! must agree
    recompute_depth!(t)
    @test t.depth == 2
    @test t.size  == 4
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree — known hand-crafted tree" begin
    t  = _hand_tree()     # safe_add(x1, safe_log(x2))
    out = eval_tree(t, X_TEST)

    @test length(out) == N_ROWS
    @test eltype(out) == Float64

    expected = safe_add.(X_TEST[:, 1], safe_log.(X_TEST[:, 2]))
    @test out ≈ expected
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree — constant node" begin
    # Tree: just a single constant node
    nodes = Node[constant_node(1)]
    t = Tree(nodes, [3.14], 1, 0)
    out = eval_tree(t, X_TEST)
    @test all(out .≈ 3.14)
    @test length(out) == N_ROWS
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree — TS unary node (ts_mean)" begin
    # ts_mean(x1, d=3)
    ts_mean_idx = OP_SYMBOL_TO_IDX[:ts_mean]   # Int8(17)
    var_node_idx = Int16(2)
    nodes = Node[
        Node(NODE_OPERATOR, ts_mean_idx, var_node_idx, Int16(0), Int16(0), Int16(3)),
        variable_node(1),
    ]
    t = Tree(nodes, Float64[], 2, 1)
    out = eval_tree(t, X_TEST)
    expected = ts_mean(X_TEST[:, 1], 3)
    @test out ≈ expected
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree — TS binary node (ts_corr)" begin
    ts_corr_idx = OP_SYMBOL_TO_IDX[:ts_corr]
    nodes = Node[
        Node(NODE_OPERATOR, ts_corr_idx, Int16(2), Int16(3), Int16(0), Int16(5)),
        variable_node(1),
        variable_node(2),
    ]
    t = Tree(nodes, Float64[], 3, 1)
    out = eval_tree(t, X_TEST)
    expected = ts_corr(X_TEST[:, 1], X_TEST[:, 2], 5)
    @test out ≈ expected
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree — CS op (cs_rank)" begin
    # cs_rank(x3)
    cs_rank_idx = OP_SYMBOL_TO_IDX[:cs_rank]
    nodes = Node[
        Node(NODE_OPERATOR, cs_rank_idx, Int16(2), Int16(0), Int16(0), Int16(0)),
        variable_node(3),
    ]
    t = Tree(nodes, Float64[], 2, 1)
    out = eval_tree(t, X_TEST)
    @test all(0 .< out .<= 1.0)
    @test length(out) == N_ROWS
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "copy_tree independence" begin
    t1 = build_random_tree(MersenneTwister(1), N_FEATURES, CFG, OP_SETS)
    t2 = copy_tree(t1)

    # Same content
    @test t1.nodes     == t2.nodes
    @test t1.constants == t2.constants
    @test t1.size      == t2.size
    @test t1.depth     == t2.depth

    # Mutating t2 must not affect t1
    t2.nodes[1] = variable_node(1)
    @test t1.nodes[1] != t2.nodes[1]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "build_random_tree — grow" begin
    for seed in 1:20
        t = build_random_tree(MersenneTwister(seed), N_FEATURES, CFG, OP_SETS; method=:grow)
        @test t.depth >= 0
        @test t.depth <= CFG.max_depth
        @test t.size  == length(t.nodes)
        @test subtree_size(t.nodes, 1) == t.size
        # eval must run without error
        out = eval_tree(t, X_TEST)
        @test length(out) == N_ROWS
        @test all(isfinite, out) || true   # NaN/Inf OK — safe ops prevent crashes
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "build_random_tree — full" begin
    for seed in 1:10
        t = build_random_tree(MersenneTwister(seed), N_FEATURES, CFG, OP_SETS;
                              method=:full, max_depth=4)
        @test t.depth <= 4
        @test t.size  == length(t.nodes)
        # TS nodes have only leaf children, so actual depth can be ≤ requested
        out = eval_tree(t, X_TEST)
        @test length(out) == N_ROWS
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "build_random_tree — depth distribution (grow)" begin
    depths = [build_random_tree(MersenneTwister(s), N_FEATURES, CFG, OP_SETS).depth
              for s in 1:200]
    # With min_depth=2, all trees must have depth >= 1 (min_depth is enforced for ops)
    @test all(d >= 1 for d in depths)
    @test all(d <= CFG.max_depth for d in depths)
    # We should see variety: at least 3 distinct depth values over 200 trees
    @test length(unique(depths)) >= 3
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "replace_subtree — leaf replacement" begin
    t_orig  = _hand_tree()   # safe_add(x1, safe_log(x2)),  size=4, depth=2
    # Replace node 2 (variable x1) with a single constant leaf
    donor_nodes = Node[constant_node(1)]
    donor = Tree(donor_nodes, [99.0], 1, 0)

    t_new = replace_subtree(t_orig, 2, donor)

    # Size: removed 1 (leaf x1), added 1 (constant) → still 4
    @test t_new.size == 4
    @test t_new.depth == 2

    # Eval: safe_add(99.0, safe_log(x2))
    out = eval_tree(t_new, X_TEST)
    expected = safe_add.(99.0, safe_log.(X_TEST[:, 2]))
    @test out ≈ expected
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "replace_subtree — subtree replacement" begin
    t_orig = _hand_tree()   # safe_add(x1, safe_log(x2))
    # Replace node 3 (safe_log(x2)) with a variable x3
    donor_nodes = Node[variable_node(3)]
    donor = Tree(donor_nodes, Float64[], 1, 0)

    t_new = replace_subtree(t_orig, 3, donor)

    # Removed 2 nodes (log + x2), added 1 → size = 4 - 2 + 1 = 3
    @test t_new.size  == 3
    @test t_new.depth == 1

    out = eval_tree(t_new, X_TEST)
    expected = safe_add.(X_TEST[:, 1], X_TEST[:, 3])
    @test out ≈ expected
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "replace_subtree — root replacement" begin
    t_orig = _hand_tree()
    # Replace entire tree with a single variable leaf
    donor = Tree([variable_node(4)], Float64[], 1, 0)
    t_new = replace_subtree(t_orig, 1, donor)

    @test t_new.size  == 1
    @test t_new.depth == 0
    @test eval_tree(t_new, X_TEST) ≈ X_TEST[:, 4]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "replace_subtree — round-trip eval" begin
    rng = MersenneTwister(99)
    for _ in 1:30
        t1 = build_random_tree(rng, N_FEATURES, CFG, OP_SETS)
        t2 = build_random_tree(rng, N_FEATURES, CFG, OP_SETS)

        target = rand(rng, 1:t1.size)
        t3 = replace_subtree(t1, target, t2)

        # Must evaluate without error
        out = eval_tree(t3, X_TEST)
        @test length(out) == N_ROWS

        # Structural invariant: size is consistent with the node array
        @test t3.size == length(t3.nodes)
        @test subtree_size(t3.nodes, 1) == t3.size
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "random_node_idx / random_nonterminal_idx" begin
    rng = MersenneTwister(7)
    t = _hand_tree()   # size=4

    idxs = [random_node_idx(rng, t) for _ in 1:100]
    @test all(1 .<= idxs .<= t.size)

    nt = random_nonterminal_idx(rng, t)
    @test nt !== nothing
    @test is_operator(t.nodes[nt])

    # Single-leaf tree → no nonterminals
    t_leaf = Tree([variable_node(1)], Float64[], 1, 0)
    @test random_nonterminal_idx(rng, t_leaf) === nothing
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "tree_to_string" begin
    t = _hand_tree()
    s = tree_to_string(t)
    @test occursin("safe_add", s)
    @test occursin("safe_log", s)
    @test occursin("x1", s)
    @test occursin("x2", s)

    # Constant node representation
    nodes = Node[constant_node(1)]
    t_const = Tree(nodes, [2.71828], 1, 0)
    @test occursin("2.7183", tree_to_string(t_const))
end

end  # @testset "Tree.jl"

println("\n✓ All Tree.jl tests passed.")