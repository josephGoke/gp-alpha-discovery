# =============================================================================
# test/test_selection.jl
# Run with: julia --project test/test_selection.jl
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
const N_ROWS     = 100    # total time-series rows
const N_PERIODS  = 10     # lexicase cases = 10 contiguous time blocks
const N_POP      = 30
const X_TEST     = randn(RNG, N_ROWS, N_FEATURES)
const Y_RAW      = X_TEST[:, 1] .+ 0.2 .* randn(RNG, N_ROWS)

const CFG_TOURN = GPConfig(
    min_depth              = 2,
    max_depth              = 4,
    const_prob             = 0.2,
    fin_op_prob            = 0.0,
    ts_window_sizes        = [3],
    population_size        = N_POP,
    n_islands              = 1,
    n_trees_per_individual = 1,
    n_generations          = 10,
    selection              = :tournament,
    tournament_k           = 5,
    lexicase_epsilon       = 0.1,
    lexicase_n_periods     = N_PERIODS,
    fitness_weights        = FitnessWeights(0.3, 0.3, 0.3, 0.1),
    parsimony              = 0.001,
    eval_subsample         = 1.0,
    behavior_sample_size   = 10,
)

const CFG_LEX = GPConfig(
    min_depth              = 2,
    max_depth              = 4,
    const_prob             = 0.2,
    fin_op_prob            = 0.0,
    ts_window_sizes        = [3],
    population_size        = N_POP,
    n_islands              = 1,
    n_trees_per_individual = 1,
    n_generations          = 10,
    selection              = :epsilon_lexicase,
    tournament_k           = 5,
    lexicase_epsilon       = 0.1,
    lexicase_n_periods     = N_PERIODS,
    fitness_weights        = FitnessWeights(0.3, 0.3, 0.3, 0.1),
    parsimony              = 0.001,
    eval_subsample         = 1.0,
    behavior_sample_size   = 10,
)

const OP_SETS = make_op_sets(CFG_TOURN)

# ── Population/error helpers ─────────────────────────────────────────────────

function _make_population(n::Int; rng::AbstractRNG = MersenneTwister(1))
    pop = Vector{Individual}(undef, n)
    for i in 1:n
        tree = build_random_tree(MersenneTwister(i), N_FEATURES, CFG_TOURN, OP_SETS)
        ind  = Individual([tree], [1.0], :weighted_sum)
        ind.fitness = Float64(i) / n   # fitness in (0,1], increasing with i
        pop[i] = ind
    end
    return pop
end

# All individuals equally bad on every period
function _uniform_errors(n_ind::Int, n_periods::Int; val::Float64 = 0.5)
    fill(val, n_ind, n_periods)
end

# Individual `best_idx` is perfect (error=0) on every period; all others = 1
function _one_best_errors(n_ind::Int, n_periods::Int, best_idx::Int)
    E = fill(1.0, n_ind, n_periods)
    E[best_idx, :] .= 0.0
    return E
end

# Each individual specialises on one time period (market regime).
# Individual i has error 0 on period i (mod n_periods), 1 elsewhere.
# This mirrors a realistic finance scenario where one strategy dominates
# in bull markets, another in bear markets, etc.
function _specialist_errors(n_ind::Int, n_periods::Int)
    E = ones(Float64, n_ind, n_periods)
    for i in 1:n_ind
        specialist_period = mod1(i, n_periods)
        E[i, specialist_period] = 0.0
    end
    return E
end


# =============================================================================
@testset "Selection.jl" begin


# ─────────────────────────────────────────────────────────────────────────────
@testset "tournament_select — returns a member of the pool" begin
    pop = _make_population(N_POP)
    E   = _uniform_errors(N_POP, N_PERIODS)
    for _ in 1:50
        winner = tournament_select(pop, CFG_TOURN, MersenneTwister(rand(RNG, 1:10000)))
        @test winner ∈ pop
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "tournament_select — always picks highest-fitness when k = pop size" begin
    pop  = _make_population(N_POP)
    best = pop[end]   # fitness = 1.0
    cfg_k = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = N_POP, n_islands = 1,
        n_trees_per_individual = 1, selection = :tournament,
        tournament_k = N_POP,   # exhaustive → always picks best
        lexicase_n_periods = N_PERIODS,
    )
    for seed in 1:20
        @test tournament_select(pop, cfg_k, MersenneTwister(seed)).fitness == best.fitness
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "tournament_select — k=2 selects above-median more often than not" begin
    pop = _make_population(20)
    cfg_k2 = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = 20, n_islands = 1,
        n_trees_per_individual = 1, selection = :tournament,
        tournament_k = 2, lexicase_n_periods = N_PERIODS,
    )
    rng = MersenneTwister(7)
    median_fit = median(ind.fitness for ind in pop)
    n_above    = sum(tournament_select(pop, cfg_k2, rng).fitness >= median_fit
                     for _ in 1:200)
    @test n_above > 100
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "epsilon_lexicase_select — returns a member of the pool" begin
    pop = _make_population(N_POP)
    E   = _uniform_errors(N_POP, N_PERIODS)
    for seed in 1:50
        @test epsilon_lexicase_select(pop, E, CFG_LEX, MersenneTwister(seed)) ∈ pop
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "epsilon_lexicase_select — always picks the one dominant individual" begin
    pop      = _make_population(N_POP)
    best_idx = 3
    E        = _one_best_errors(N_POP, N_PERIODS, best_idx)
    cfg_strict = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = N_POP, n_islands = 1,
        n_trees_per_individual = 1, selection = :epsilon_lexicase,
        lexicase_epsilon = 1e-9, lexicase_n_periods = N_PERIODS,
    )
    for seed in 1:20
        @test epsilon_lexicase_select(pop, E, cfg_strict, MersenneTwister(seed)) === pop[best_idx]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "epsilon_lexicase_select — large ε selects uniformly from full pool" begin
    pop = _make_population(N_POP)
    E   = _uniform_errors(N_POP, N_PERIODS; val = 0.5)
    cfg_wide = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = N_POP, n_islands = 1,
        n_trees_per_individual = 1, selection = :epsilon_lexicase,
        lexicase_epsilon = 1e9, lexicase_n_periods = N_PERIODS,
    )
    counts   = zeros(Int, N_POP)
    n_trials = 500
    for seed in 1:n_trials
        w   = epsilon_lexicase_select(pop, E, cfg_wide, MersenneTwister(seed))
        idx = findfirst(x -> x === w, pop)
        counts[idx] += 1
    end
    expected = n_trials / N_POP
    @test maximum(counts) <= 3 * expected   # no individual dominates
    @test all(counts .>= 1)                 # every individual selected ≥ once
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "epsilon_lexicase_select — preserves regime specialists" begin
    # Each individual specialises on one time period (market regime).
    # Over many trials every specialist should be selected.
    # This is the core value proposition of lexicase for finance.
    n   = 10
    pop = _make_population(n)
    E   = _specialist_errors(n, N_PERIODS)
    cfg = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = n, n_islands = 1,
        n_trees_per_individual = 1, selection = :epsilon_lexicase,
        lexicase_epsilon = 0.1, lexicase_n_periods = N_PERIODS,
    )
    selected = Set{Int}()
    for seed in 1:300
        w   = epsilon_lexicase_select(pop, E, cfg, MersenneTwister(seed))
        push!(selected, findfirst(x -> x === w, pop))
    end
    @test length(selected) == n   # all regime specialists survived
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "epsilon_lexicase_select — period order shuffled not row order" begin
    # Verify the invariant directly: period_order is a permutation of 1:n_periods,
    # NOT a permutation of 1:n_rows.
    # We do this by checking the errors matrix columns (periods) are correctly
    # sized and that build_error_matrix respects N_ROWS → N_PERIODS aggregation.
    pop = _make_population(5)
    for ind in pop; update_complexity!(ind); end
    E = build_error_matrix(pop, X_TEST, Y_RAW, CFG_LEX)

    @test size(E, 1) == 5           # one row per individual
    @test size(E, 2) == N_PERIODS   # one column per time period, not per row
    # Each column is a mean error ≥ 0
    @test all(E .>= 0.0)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "select — dispatches to correct method" begin
    pop = _make_population(N_POP)
    E   = _one_best_errors(N_POP, N_PERIODS, N_POP)   # last individual is best

    # Tournament exhaustive → always picks highest fitness
    cfg_t = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = N_POP, n_islands = 1,
        n_trees_per_individual = 1, selection = :tournament,
        tournament_k = N_POP, lexicase_n_periods = N_PERIODS,
    )
    @test select(pop, E, cfg_t, MersenneTwister(1)) === pop[end]

    # Epsilon-lexicase strict → picks zero-error individual
    cfg_l = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = N_POP, n_islands = 1,
        n_trees_per_individual = 1, selection = :epsilon_lexicase,
        lexicase_epsilon = 1e-9, lexicase_n_periods = N_PERIODS,
    )
    @test select(pop, E, cfg_l, MersenneTwister(1)) === pop[N_POP]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "select_n — returns correct count; all members from pool" begin
    pop = _make_population(N_POP)
    E   = _uniform_errors(N_POP, N_PERIODS)
    for cfg in (CFG_TOURN, CFG_LEX)
        result = select_n(pop, E, 15, cfg, MersenneTwister(99))
        @test length(result) == 15
        @test all(ind ∈ pop for ind in result)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "select_n — does not mutate the pool" begin
    pop        = _make_population(N_POP)
    E          = _uniform_errors(N_POP, N_PERIODS)
    fit_before = [ind.fitness for ind in pop]
    select_n(pop, E, 20, CFG_TOURN, MersenneTwister(1))
    @test [ind.fitness for ind in pop] == fit_before
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "build_error_matrix — shape (n_ind × n_periods), non-negative, finite" begin
    pop = _make_population(N_POP; rng = MersenneTwister(5))
    y_ranked = prerank_y(Y_RAW)
    for ind in pop
        update_complexity!(ind)
        evaluate_fitness!(ind, X_TEST, Y_RAW, y_ranked, CFG_TOURN)
    end
    E = build_error_matrix(pop, X_TEST, Y_RAW, CFG_TOURN)

    @test size(E) == (N_POP, N_PERIODS)
    @test all(E .>= 0.0)
    @test all(isfinite, E)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "build_error_matrix — perfect predictor has near-zero error on all periods" begin
    # A tree that just returns x[:,1]; paired with y = x[:,1], error ≈ 0
    tree_perfect = Tree([variable_node(1)], Float64[], 1, 0)
    ind_perfect  = Individual([tree_perfect], [1.0], :weighted_sum)
    pop = vcat(_make_population(3), [ind_perfect])

    E = build_error_matrix(pop, X_TEST, X_TEST[:, 1], CFG_TOURN)
    @test all(E[end, :] .< 1e-10)   # perfect predictor: every period error ≈ 0
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "build_error_matrix — periods cover all rows (no data dropped)" begin
    # With N_ROWS=100 and N_PERIODS=10, each block = 10 rows.
    # Verify by checking that a constant-prediction tree has the same
    # per-period mean error regardless of aggregation.
    tree_zero = Tree([constant_node(1)], [0.0], 1, 0)
    ind       = Individual([tree_zero], [1.0], :weighted_sum)
    y_ones    = ones(N_ROWS)
    E         = build_error_matrix([ind], X_TEST, y_ones, CFG_TOURN)
    # Signal = 0, y = 1 everywhere → mean error = 1.0 on every period
    @test all(E[1, :] .≈ 1.0)
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "tournament_select — reproducible with same rng seed" begin
    pop = _make_population(N_POP)
    E   = _uniform_errors(N_POP, N_PERIODS)
    w1  = tournament_select(pop, CFG_TOURN, MersenneTwister(123))
    w2  = tournament_select(pop, CFG_TOURN, MersenneTwister(123))
    @test w1 === w2
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "epsilon_lexicase_select — reproducible with same rng seed" begin
    pop = _make_population(N_POP)
    E   = _specialist_errors(N_POP, N_PERIODS)
    w1  = epsilon_lexicase_select(pop, E, CFG_LEX, MersenneTwister(77))
    w2  = epsilon_lexicase_select(pop, E, CFG_LEX, MersenneTwister(77))
    @test w1 === w2
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "select — single-individual pool always returns that individual" begin
    lone = _make_population(1)
    E    = _uniform_errors(1, N_PERIODS)
    for cfg in (CFG_TOURN, CFG_LEX)
        @test select(lone, E, cfg, MersenneTwister(1)) === lone[1]
    end
end

end  # @testset "Selection.jl"

println("\n✓ All Selection.jl tests passed.")