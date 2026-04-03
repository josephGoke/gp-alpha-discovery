# =============================================================================
# test/test_constants.jl
# Run with: julia --project test/test_constants.jl
# =============================================================================

using Test
using Random
using Statistics

include("../src/Genetic.jl")
using .Genetic

# =============================================================================
# Shared fixtures
# =============================================================================

const RNG        = MersenneTwister(42)
const N_FEATURES = 5
const N_ROWS     = 200
const X_TEST     = randn(RNG, N_ROWS, N_FEATURES)
const Y_RAW      = X_TEST[:, 1] .+ 0.2 .* randn(RNG, N_ROWS)   # x1 + noise
const Y_RANKED   = prerank_y(Y_RAW)

const CFG = GPConfig(
    min_depth              = 2,
    max_depth              = 4,
    const_prob             = 0.3,
    fin_op_prob            = 0.0,
    ts_window_sizes        = [3, 5],
    population_size        = 20,
    n_islands              = 1,
    n_trees_per_individual = 2,
    fitness_weights        = FitnessWeights(0.3, 0.3, 0.3, 0.1),
    n_generations          = 10,
    parsimony              = 0.001,
    eval_subsample         = 0.5,
    behavior_sample_size   = 20,
    const_opt_method       = :bfgs,
    const_opt_every_n_gens = 2,
)
const OP_SETS = make_op_sets(CFG)

# Helper: c * x_col  (scale-only; PPS IS sensitive to scale unlike raw Pearson)
function _linear_tree(col::Int = 1, c_init::Float64 = 0.1)
    nodes = [
        op_node(Int8(12), Int16(2), Int16(3)),   # safe_mul, idx=12
        variable_node(col),
        constant_node(1),
    ]
    Tree(nodes, [c_init], 3, 1)
end

# Helper: safe_add(x1, x2)  — no constants
function _no_const_tree()
    nodes = [
        op_node(Int8(10), Int16(2), Int16(3)),   # safe_add, idx=10
        variable_node(1),
        variable_node(2),
    ]
    Tree(nodes, Float64[], 3, 1)
end

# Helper: ts_mean(c*x1, d=5)  — contains a TS op
function _ts_tree(c_init::Float64 = 0.5)
    nodes = [
        Node(NODE_OPERATOR, IDX_TS_UNARY_START, Int16(2), Int16(0), Int16(0), Int16(5)),
        op_node(Int8(12), Int16(3), Int16(4)),
        variable_node(1),
        constant_node(1),
    ]
    Tree(nodes, [c_init], 4, 2)
end

# Helper: x_signal + c * x_noise
#
# PPS (like IC) is maximised when noise is suppressed (c → 0).
# Starting at c_init=10 gives the optimizer a clear task.
# Layout:
#   nodes[1] = safe_add([2],[3])
#   nodes[2] = variable x_signal
#   nodes[3] = safe_mul([4],[5])
#   nodes[4] = constant[1] = c_init
#   nodes[5] = variable x_noise
function _signal_plus_noise_tree(signal_col::Int = 1, noise_col::Int = 3,
                                   c_init::Float64 = 10.0)
    nodes = [
        op_node(Int8(10), Int16(2), Int16(3)),
        variable_node(signal_col),
        op_node(Int8(12), Int16(4), Int16(5)),
        constant_node(1),
        variable_node(noise_col),
    ]
    Tree(nodes, [c_init], 5, 2)
end

# Helper: individual wrapping two signal+noise trees
function _noisy_ind(; c_init::Float64 = 10.0)
    t1 = _signal_plus_noise_tree(1, 3, c_init)
    t2 = _signal_plus_noise_tree(1, 4, c_init)
    Individual([t1, t2], [0.5, 0.5], :weighted_sum)
end


# =============================================================================
@testset "Constants.jl" begin

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree_with_consts — matches eval_tree with original constants" begin
    rng = MersenneTwister(1)
    for _ in 1:20
        tree = build_random_tree(rng, N_FEATURES, CFG, OP_SETS)
        expected = eval_tree(tree, X_TEST)
        got      = eval_tree_with_consts(tree, X_TEST, tree.constants)
        @test got ≈ expected
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree_with_consts — reflects injected constants without mutating tree" begin
    tree = _linear_tree(1, 1.0)

    out_1 = eval_tree_with_consts(tree, X_TEST, [1.0])
    out_2 = eval_tree_with_consts(tree, X_TEST, [2.0])
    out_0 = eval_tree_with_consts(tree, X_TEST, [0.0])

    @test out_2 ≈ 2.0 .* out_1  atol=1e-8
    @test all(abs.(out_0) .< 1e-8)
    @test tree.constants[1] ≈ 1.0   # tree itself unchanged
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree_with_consts — output length = nrows(X)" begin
    rng = MersenneTwister(2)
    for _ in 1:10
        tree = build_random_tree(rng, N_FEATURES, CFG, OP_SETS)
        out  = eval_tree_with_consts(tree, X_TEST, tree.constants)
        @test length(out) == N_ROWS
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "eval_tree_with_consts — empty tree returns zero vector" begin
    empty_tree = Tree()
    out = eval_tree_with_consts(empty_tree, X_TEST, Float64[])
    @test all(out .== 0.0)
    @test length(out) == N_ROWS
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "pps — used as loss: boundary behaviour" begin
    n = 100
    x = collect(Float64, 1:n)
    y = collect(Float64, 1:n)
    yr = prerank_y(y)

    # Perfect positive correlation → PPS ≈ 1
    @test pps(x, y, yr) ≈ 1.0  atol=1e-6

    # Perfect negative correlation → PPS ≈ -1
    @test pps(-x, y, yr) ≈ -1.0  atol=1e-6

    # Constant signal → PPS ≈ 0
    @test abs(pps(ones(n), y, yr)) < 1e-6

    # Empty → 0
    @test pps(Float64[], Float64[], Float64[]) == 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_constants! — improves PPS on noisy signal" begin
    # Tree: x1 + c*x3,  c_init = 10.0
    # Y = x1 + 0.2*noise  →  optimal c ≈ 0 (suppress decorrelated noise column)
    tree = _signal_plus_noise_tree(1, 3, 10.0)

    pps_before = pps(eval_tree(tree, X_TEST), Y_RAW, Y_RANKED)
    updated    = optimize_constants!(tree, X_TEST, Y_RAW, Y_RANKED, CFG)
    pps_after  = pps(eval_tree(tree, X_TEST), Y_RAW, Y_RANKED)

    @test updated                             # optimizer ran
    @test pps_after > pps_before              # PPS improved
    @test abs(tree.constants[1]) < 5.0        # c moved toward 0 from 10
    @test pps_after > 0.7                     # strong signal recovered
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_constants! — optimizer improves PPS for negatively correlated target" begin
    # Tree: x1 + c*x3,  target = -x1 + noise,  c_init = 10.0
    #
    # Optimizer minimizes -PPS, i.e. maximizes PPS.
    # With y_neg ≈ -x1, PPS(x1 + c*x3, y_neg) is negative everywhere
    # (signal and target are negatively correlated). The optimizer finds the
    # value of c that makes PPS as large as possible (least negative → toward 0),
    # which may mean adding noise to dilute the anti-signal rather than
    # suppressing it. That is correct optimizer behaviour.
    #
    # The invariant we can always assert: pps_after >= pps_before
    # (the optimizer never makes the objective worse).
    y_neg  = -X_TEST[:, 1] .+ 0.2 .* randn(RNG, N_ROWS)
    yr_neg = prerank_y(y_neg)
    tree   = _signal_plus_noise_tree(1, 3, 10.0)

    pps_before = pps(eval_tree(tree, X_TEST), y_neg, yr_neg)
    optimize_constants!(tree, X_TEST, y_neg, yr_neg, CFG)
    pps_after  = pps(eval_tree(tree, X_TEST), y_neg, yr_neg)

    # Optimizer must not make PPS worse (monotone improvement guarantee)
    @test pps_after >= pps_before - 1e-6
    # Constants changed (optimizer did something)
    @test tree.constants[1] != 10.0
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_constants! — no constants → returns false, tree unchanged" begin
    tree = _no_const_tree()
    nodes_before = copy(tree.nodes)

    updated = optimize_constants!(tree, X_TEST, Y_RAW, Y_RANKED, CFG)

    @test !updated
    @test tree.nodes == nodes_before
    @test isempty(tree.constants)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_constants! — tree still evaluable after optimization" begin
    for seed in 1:20
        tree = build_random_tree(MersenneTwister(seed), N_FEATURES, CFG, OP_SETS)
        optimize_constants!(tree, X_TEST, Y_RAW, Y_RANKED, CFG)
        out = eval_tree(tree, X_TEST)
        @test out isa Vector{Float64}
        @test length(out) == N_ROWS
        @test !all(isnan, out)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_constants! — does not mutate tree structure" begin
    tree         = _signal_plus_noise_tree(1, 3, 10.0)
    size_before  = tree.size
    depth_before = tree.depth
    nodes_copy   = copy(tree.nodes)

    optimize_constants!(tree, X_TEST, Y_RAW, Y_RANKED, CFG)

    @test tree.size  == size_before
    @test tree.depth == depth_before
    for i in 1:tree.size
        nb = nodes_copy[i];  na = tree.nodes[i]
        @test na.node_type == nb.node_type
        @test na.op_idx    == nb.op_idx
        @test na.left      == nb.left
        @test na.right     == nb.right
        @test na.feature   == nb.feature
        @test na.const_idx == nb.const_idx
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_constants! — works on TS trees" begin
    cfg_ts = GPConfig(
        min_depth              = 2,
        max_depth              = 4,
        const_prob             = 0.3,
        fin_op_prob            = 1.0,
        ts_window_sizes        = [5],
        population_size        = 20,
        n_islands              = 1,
        n_trees_per_individual = 1,
        const_opt_method       = :nelder_mead,
        const_opt_every_n_gens = 1,
    )
    op_ts = make_op_sets(cfg_ts)

    for seed in 1:5
        tree = build_random_tree(MersenneTwister(seed), N_FEATURES, cfg_ts, op_ts)
        isempty(tree.constants) && continue

        # Should complete without throwing (warnings are acceptable)
        updated = optimize_constants!(tree, X_TEST, Y_RAW, Y_RANKED, cfg_ts)
        @test updated isa Bool

        out = eval_tree(tree, X_TEST)
        @test out isa Vector{Float64}
        @test length(out) == N_ROWS
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_individual_constants! — PPS improves on all trees" begin
    ind = _noisy_ind(c_init = 10.0)

    pps_before_t1 = pps(eval_tree(ind.trees[1], X_TEST), Y_RAW, Y_RANKED)
    pps_before_t2 = pps(eval_tree(ind.trees[2], X_TEST), Y_RAW, Y_RANKED)

    optimize_individual_constants!(ind, X_TEST, Y_RAW, Y_RANKED, CFG)

    pps_after_t1 = pps(eval_tree(ind.trees[1], X_TEST), Y_RAW, Y_RANKED)
    pps_after_t2 = pps(eval_tree(ind.trees[2], X_TEST), Y_RAW, Y_RANKED)

    @test pps_after_t1 > pps_before_t1
    @test pps_after_t2 > pps_before_t2
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_population_constants! — skips non-scheduled generations" begin
    pop = [_noisy_ind(c_init = 10.0) for _ in 1:10]
    c0  = [ind.trees[1].constants[1] for ind in pop]

    optimize_population_constants!(pop, X_TEST, Y_RAW, Y_RANKED, CFG, 1)  # gen=1 skipped

    c1 = [ind.trees[1].constants[1] for ind in pop]
    @test c0 == c1
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_population_constants! — runs on scheduled generation" begin
    pop = [_noisy_ind(c_init = 10.0) for _ in 1:10]
    for ind in pop; ind.fitness = randn(RNG); end

    c0 = [ind.trees[1].constants[1] for ind in pop]
    optimize_population_constants!(pop, X_TEST, Y_RAW, Y_RANKED, CFG, 2)  # gen=2 runs
    c1 = [ind.trees[1].constants[1] for ind in pop]

    @test sum(c0[i] != c1[i] for i in 1:length(pop)) > 0
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_population_constants! — only elite_count individuals updated" begin
    n_pop = CFG.population_size
    pop   = [_noisy_ind(c_init = 10.0) for _ in 1:n_pop]
    for (k, ind) in enumerate(pop)
        ind.fitness = k <= CFG.elite_count ? 1.0 : -100.0
    end

    optimize_population_constants!(pop, X_TEST, Y_RAW, Y_RANKED, CFG, 2)

    n_changed = sum(ind.trees[1].constants[1] != 10.0 for ind in pop)
    @test n_changed <= CFG.elite_count
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "optimize_constants! — determinism: same inputs → same result" begin
    tree_a = _signal_plus_noise_tree(1, 3, 10.0)
    tree_b = _signal_plus_noise_tree(1, 3, 10.0)

    optimize_constants!(tree_a, X_TEST, Y_RAW, Y_RANKED, CFG)
    optimize_constants!(tree_b, X_TEST, Y_RAW, Y_RANKED, CFG)

    @test tree_a.constants ≈ tree_b.constants  atol=1e-6
end

end  # @testset "Constants.jl"

println("\n✓ All Constants.jl tests passed.")