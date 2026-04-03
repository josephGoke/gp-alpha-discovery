# =============================================================================
# test/test_fitness.jl
# Run with: julia --project test/test_fitness.jl
# =============================================================================

using Test
using Random
using Statistics
using StatsBase: tiedrank

include("../src/Types.jl")
include("../src/Functions.jl")
include("../src/Tree.jl")
include("../src/Evaluate.jl")
include("../src/Fitness.jl")

# ── Shared fixtures ───────────────────────────────────────────────────────────

const RNG        = MersenneTwister(42)
const N_FEATURES = 5
const N_ROWS     = 200
const X_TEST     = randn(RNG, N_ROWS, N_FEATURES)
const Y_RAW      = randn(RNG, N_ROWS)
const Y_RANKED   = prerank_y(Y_RAW)

const CFG = GPConfig(
    min_depth            = 2,
    max_depth            = 5,
    const_prob           = 0.2,
    fin_op_prob          = 0.3,
    ts_window_sizes      = [3, 5, 10],
    population_size      = 100,
    n_islands            = 1,
    fitness_weights      = FitnessWeights(0.3, 0.3, 0.3, 0.1),
    n_generations        = 100,
    parsimony            = 0.001,
    eval_subsample       = 0.5,
    behavior_sample_size = 20,
)

const OP_SETS = make_op_sets(CFG)

# Helper: build a random multi-tree individual
function _make_ind(combination::Symbol = :rank_average; n_trees::Int = 2)
    trees   = [build_random_tree(MersenneTwister(s), N_FEATURES, CFG, OP_SETS)
               for s in 1:n_trees]
    weights = ones(n_trees) ./ n_trees
    Individual(trees, weights, combination)
end

# Helper: single-leaf individual wrapping a known feature column
function _leaf_ind(col::Int, combination::Symbol = :rank_average)
    nodes = Node[variable_node(col)]
    t = Tree(nodes, Float64[], 1, 0)
    Individual([t], [1.0], combination)
end


# =============================================================================
@testset "Fitness.jl" begin

# ─────────────────────────────────────────────────────────────────────────────
@testset "prerank_y" begin
    y = [3.0, 1.0, 4.0, 1.0, 5.0]
    r = prerank_y(y)
    @test length(r) == 5
    @test minimum(r) >= 1.0
    @test maximum(r) == 5.0
    # Ties: both 1.0 values should share the same average rank (1.5)
    @test r[2] == r[4]
    @test r[2] ≈ 1.5
    # Monotonicity: rank(3.0) < rank(4.0) < rank(5.0)
    @test r[1] < r[3] < r[5]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "rank_ic — boundary cases" begin
    n = 100
    y_r = prerank_y(collect(Float64, 1:n))

    # Perfect positive correlation
    f_pos = collect(Float64, 1:n)
    @test rank_ic(f_pos, y_r) ≈ 1.0 atol=1e-8

    # Perfect negative correlation
    f_neg = collect(Float64, n:-1:1)
    @test rank_ic(f_neg, y_r) ≈ -1.0 atol=1e-8

    # Constant factor → 0.0 (no information)
    f_const = ones(Float64, n)
    @test rank_ic(f_const, y_r) == 0.0

    # Length mismatch → error
    @test_throws ErrorException rank_ic(ones(10), ones(9))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "rank_ic — random signal near zero" begin
    rng = MersenneTwister(7)
    y_r = prerank_y(randn(rng, 500))
    # Independent random factor should have IC close to 0
    ics = [rank_ic(randn(rng, 500), y_r) for _ in 1:50]
    @test mean(abs.(ics)) < 0.15   # loose bound; exact value is random
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "rank_ic — monotone transform invariance" begin
    # rank is invariant to monotone transforms
    y_r  = prerank_y(Y_RAW)
    f    = X_TEST[:, 1]
    ic1  = rank_ic(f, y_r)
    ic2  = rank_ic(exp.(f), y_r)   # exp is monotone → same rank order
    @test ic1 ≈ ic2 atol=1e-6
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "icir — basic properties" begin
    # Perfectly consistent signal → std(IC) = 0 → our safety returns 0.0
    n = 200
    y_raw_lin = collect(Float64, 1:n)
    factor    = collect(Float64, 1:n)
    ir = icir(factor, y_raw_lin, 10)
    @test ir == 0.0  # std IC is exactly 0

    # Zero-IC signal → icir ≈ 0
    y_rand = randn(MersenneTwister(1), n)
    f_rand = randn(MersenneTwister(2), n)
    ir_rand = icir(f_rand, y_rand, 10)
    @test abs(ir_rand) < 3.0   # may not be exactly 0 with small windows

    # Edge cases
    @test icir(factor, y_raw_lin, 1)  == 0.0   # n_periods < 2
    @test icir(factor, y_raw_lin, 0)  == 0.0
    @test icir(ones(4), ones(4), 10)  == 0.0   # period_size < 2
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "current_novelty_weight — annealing schedule" begin
    # novelty weight is stored in fitness_weights.novelty
    cfg = GPConfig(
        fitness_weights = FitnessWeights(0.0, 0.0, 0.0, 0.3),
        n_generations   = 100,
        population_size = 100,
        n_islands       = 1,
    )

    # Before anneal start (gen ≤ 75): full weight
    @test current_novelty_weight(cfg, 1)   == 0.3
    @test current_novelty_weight(cfg, 75)  == 0.3

    # At run end: zero
    @test current_novelty_weight(cfg, 100) ≈ 0.0 atol=1e-10

    # Midpoint of anneal window (75–100): half weight
    mid = current_novelty_weight(cfg, 87)
    @test 0.0 < mid < 0.3

    # Zero novelty weight: always returns 0
    cfg0 = GPConfig(
        fitness_weights = FitnessWeights(0.0, 0.0, 0.0, 0.0),
        n_generations   = 100,
        population_size = 100,
        n_islands       = 1,
    )
    @test current_novelty_weight(cfg0, 1) == 0.0

    # Beyond n_generations: clamped to 0
    @test current_novelty_weight(cfg, 200) ≈ 0.0 atol=1e-10
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "compute_fitness — PPS dominates, parsimony penalises" begin
    # Use a config with pps=1.0 so assertions are straightforward
    cfg_pps = GPConfig(
        fitness_weights  = FitnessWeights(1.0, 0.0, 0.0, 0.0),
        parsimony        = 0.001,
        population_size  = 100,
        n_islands        = 1,
    )

    y_raw  = collect(Float64, 1:N_ROWS)
    y_r    = prerank_y(y_raw)
    signal = collect(Float64, 1:N_ROWS)   # perfect correlation → pps = 1.0

    f0 = compute_fitness(signal, y_raw, y_r, 0,   cfg_pps)   # no complexity
    f1 = compute_fitness(signal, y_raw, y_r, 100, cfg_pps)   # 100-node tree
    @test f0 > f1                                             # parsimony hurts
    @test f0 ≈ 1.0 atol=1e-6
    @test f1 ≈ 1.0 - cfg_pps.parsimony * 100 atol=1e-6

    # Novelty weight shifts the balance
    cfg_nw = GPConfig(
        fitness_weights  = FitnessWeights(0.7, 0.0, 0.0, 0.3),
        parsimony        = 0.0,
        population_size  = 100,
        n_islands        = 1,
    )
    fn = compute_fitness(signal, y_raw, y_r, 0, cfg_nw; novelty=0.5)
    @test fn ≈ 0.7 * 1.0 + 0.3 * 0.5 atol=1e-6
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "extract_constants / inject_constants!" begin
    nodes = Node[
        constant_node(1),
    ]
    t = Tree(nodes, [2.718], 1, 0)

    c = extract_constants(t)
    @test c == [2.718]

    # Mutating extracted copy does not affect tree
    c[1] = 99.0
    @test t.constants[1] ≈ 2.718

    # inject_constants! writes back
    inject_constants!(t, [3.14])
    @test t.constants[1] ≈ 3.14

    # Length mismatch errors
    @test_throws ErrorException inject_constants!(t, [1.0, 2.0])
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "update_complexity!" begin
    ind = _make_ind()
    old_c = ind.complexity
    # Complexity should equal sum of tree sizes
    expected = sum(t.size for t in ind.trees)
    @test ind.complexity == expected

    # After manually growing a tree, update_complexity! catches the change
    push!(ind.trees[1].nodes, variable_node(1))
    ind.trees[1].size += 1
    update_complexity!(ind)
    @test ind.complexity == old_c + 1
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "combine_trees — :weighted_sum" begin
    ind = _leaf_ind(1, :weighted_sum)
    # Single tree of weight 1.0 → exactly X[:,1]
    out = combine_trees(ind, X_TEST)
    @test out ≈ X_TEST[:, 1]
    @test length(out) == N_ROWS
end

@testset "combine_trees — :rank_average" begin
    ind = _leaf_ind(1, :rank_average)
    out = combine_trees(ind, X_TEST)
    # Result is cs_rank of X[:,1] (single tree)
    @test out ≈ cs_rank(X_TEST[:, 1])
    @test all(0 .< out .<= 1.0)
end

@testset "combine_trees — :vote" begin
    # Two trees: x1 and safe_neg(x1). Votes cancel → all-zero signal
    t1_nodes = Node[variable_node(1)]
    t2_nodes = Node[
        Node(NODE_OPERATOR, OP_SYMBOL_TO_IDX[:safe_neg], Int16(2), Int16(0),
             Int16(0), Int16(0)),
        variable_node(1),
    ]
    t1 = Tree(t1_nodes, Float64[], 1, 0)
    t2 = Tree(t2_nodes, Float64[], 2, 1)
    ind = Individual([t1, t2], [1.0, 1.0], :vote)
    out = combine_trees(ind, X_TEST)
    # Every vote cancels: sign(x1) + sign(-x1) = 0 → sign(0) = 0
    @test all(out .== 0.0)
end

@testset "combine_trees — empty trees" begin
    ind = Individual(Tree[], Float64[], :rank_average,
                     -Inf, 0, 0, Float32[], false)
    out = combine_trees(ind, X_TEST)
    @test all(out .== 0.0)
    @test length(out) == N_ROWS
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "evaluate_fitness! — updates ind.fitness" begin
    ind = _leaf_ind(1)   # signal = cs_rank(X[:,1])
    @test ind.fitness == -Inf   # unevaluated

    y_r = prerank_y(Y_RAW)
    f   = evaluate_fitness!(ind, X_TEST, Y_RAW, y_r, CFG)

    @test ind.fitness == f
    @test isfinite(f)
    @test -2.0 <= f <= 2.0   # IC-based, should be bounded
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "update_behavior!" begin
    ind      = _leaf_ind(1)
    beh_rows = collect(1:CFG.behavior_sample_size)
    update_behavior!(ind, X_TEST, beh_rows)

    @test length(ind.behavior) == CFG.behavior_sample_size
    @test eltype(ind.behavior) == Float32
    # Content should match cs_rank(X_TEST[beh_rows, 1])
    expected = Float32.(cs_rank(X_TEST[beh_rows, 1]))
    @test ind.behavior ≈ expected
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "evaluate_population! — all individuals evaluated" begin
    rng      = MersenneTwister(10)
    pop      = [_make_ind() for _ in 1:20]
    beh_rows = randperm(rng, N_ROWS)[1:CFG.behavior_sample_size]

    # All start at -Inf
    @test all(ind.fitness == -Inf for ind in pop)

    evaluate_population!(pop, X_TEST, Y_RAW, beh_rows, CFG, rng, 1)

    # All must have finite fitness after evaluation
    @test all(isfinite(ind.fitness) for ind in pop)

    # Behavior fingerprints populated
    @test all(length(ind.behavior) == CFG.behavior_sample_size for ind in pop)
    @test all(eltype(ind.behavior) == Float32 for ind in pop)

    # Complexity is consistent with tree sizes
    for ind in pop
        expected_c = sum(t.size for t in ind.trees)
        @test ind.complexity == expected_c
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "evaluate_population! — determinism (same seed → same fitness)" begin
    rng1 = MersenneTwister(99)
    rng2 = MersenneTwister(99)
    beh_rows = collect(1:CFG.behavior_sample_size)

    pop1 = [_leaf_ind(1) for _ in 1:10]
    pop2 = [_leaf_ind(1) for _ in 1:10]

    evaluate_population!(pop1, X_TEST, Y_RAW, beh_rows, CFG, rng1, 1)
    evaluate_population!(pop2, X_TEST, Y_RAW, beh_rows, CFG, rng2, 1)

    @test all(pop1[i].fitness == pop2[i].fitness for i in 1:10)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "evaluate_population! — eval_subsample < 1 still produces valid fitness" begin
    cfg_sub = GPConfig(
        eval_subsample       = 0.3,
        population_size      = 50,
        n_islands            = 1,
        min_depth            = 2,
        max_depth            = 5,
        ts_window_sizes      = [3, 5],
        behavior_sample_size = 20,
    )
    rng      = MersenneTwister(5)
    beh_rows = collect(1:cfg_sub.behavior_sample_size)
    pop      = [_leaf_ind(1) for _ in 1:10]
    evaluate_population!(pop, X_TEST, Y_RAW, beh_rows, cfg_sub, rng, 1)
    @test all(isfinite(ind.fitness) for ind in pop)
end

end  # @testset "Fitness.jl"

println("\n✓ All Fitness.jl tests passed.")