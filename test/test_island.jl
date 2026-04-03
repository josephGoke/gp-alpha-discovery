# =============================================================================
# test/test_island.jl
# Run with: julia --project test/test_island.jl
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
const N_FEATURES = 4
const N_ROWS     = 80
const X_TEST     = randn(RNG, N_ROWS, N_FEATURES)
const Y_RAW      = X_TEST[:, 1] .+ 0.2 .* randn(RNG, N_ROWS)
const Y_RANKED   = prerank_y(Y_RAW)

# Small config for fast tests
const CFG = GPConfig(
    population_size        = 20,
    n_islands              = 4,          # 5 individuals per island
    n_trees_per_individual = 1,
    min_depth              = 2,
    max_depth              = 4,
    const_prob             = 0.2,
    fin_op_prob            = 0.0,
    ts_window_sizes        = [3],
    n_generations          = 5,
    elite_count            = 2,
    crossover_prob         = 0.8,
    mutation_prob          = MutationProbs(0.4, 0.35, 0.15, 0.1),
    selection              = :tournament,
    tournament_k           = 3,
    lexicase_n_periods     = 5,
    lexicase_epsilon       = 0.1,
    optimize_constants     = false,
    const_opt_every_n_gens = 99,   # effectively disable for tests
    migration_interval     = 2,
    migration_rate         = 0.2,
    migration_topology     = :ring,
    fitness_weights        = FitnessWeights(0.6, 0.2, 0.0, 0.2),
    parsimony              = 0.001,
    eval_subsample         = 1.0,
    behavior_sample_size   = 20,
    novelty_k              = 3,
    novelty_archive_size   = 10,
    hof_size               = 5,
    seed                   = 42,
    verbose                = false,
)

const OP_SETS = make_op_sets(CFG)

function _make_pop(n::Int = CFG.population_size; seed::Int = 1)
    rng = MersenneTwister(seed)
    pop = init_population(rng, N_FEATURES, CFG, OP_SETS)
    y_r = prerank_y(Y_RAW)
    beh = collect(1:CFG.behavior_sample_size)
    for ind in pop
        update_complexity!(ind)
        evaluate_fitness!(ind, X_TEST, Y_RAW, y_r, CFG)
        update_behavior!(ind, X_TEST, beh)
    end
    return pop
end


# =============================================================================
@testset "Island.jl + Engine.jl" begin


# ─────────────────────────────────────────────────────────────────────────────
@testset "partition_population — correct sizes and coverage" begin
    pop     = _make_pop()
    islands = partition_population(pop, CFG)

    @test length(islands) == CFG.n_islands

    island_size = div(CFG.population_size, CFG.n_islands)
    for island in islands
        @test length(island) == island_size
    end

    # Merge recovers the full population in order
    merged = merge_islands(islands)
    @test length(merged) == CFG.population_size
    for (a, b) in zip(merged, pop)
        @test a === b
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "migration_targets — ring topology" begin
    n = 4
    t = migration_targets(n, :ring, MersenneTwister(1))
    @test t == [2, 3, 4, 1]   # each island sends to the next; last wraps to first
end

@testset "migration_targets — star topology" begin
    n = 4
    t = migration_targets(n, :star, MersenneTwister(1))
    @test all(t .== 1)   # all islands target hub
end

@testset "migration_targets — random topology" begin
    n = 6
    rng = MersenneTwister(99)
    for _ in 1:20
        t = migration_targets(n, :random, rng)
        @test length(t) == n
        # No island sends to itself
        @test all(t[i] != i for i in 1:n)
        # All targets are valid island indices
        @test all(1 <= t[i] <= n for i in 1:n)
    end
end

@testset "migration_targets — single island returns [1]" begin
    @test migration_targets(1, :ring,   MersenneTwister(1)) == [1]
    @test migration_targets(1, :star,   MersenneTwister(1)) == [1]
    @test migration_targets(1, :random, MersenneTwister(1)) == [1]
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "migrate! :ring — top individuals transferred" begin
    pop     = _make_pop()
    islands = partition_population(pop, CFG)

    # Record best individual on island 1 before migration
    best_on_1 = maximum(ind.fitness for ind in islands[1])

    migrate!(islands, CFG, MersenneTwister(7))

    # Island 2 (receives from 1 on a ring) should contain an individual
    # whose fitness equals the pre-migration best of island 1
    fits_on_2 = [ind.fitness for ind in islands[2]]
    @test best_on_1 ∈ fits_on_2
end

@testset "migrate! — island sizes unchanged after migration" begin
    pop     = _make_pop()
    islands = partition_population(pop, CFG)
    sizes_before = [length(i) for i in islands]

    migrate!(islands, CFG, MersenneTwister(3))

    @test [length(i) for i in islands] == sizes_before
end

@testset "migrate! — sending island keeps its best individual" begin
    pop     = _make_pop()
    islands = partition_population(pop, CFG)

    # Best individual on island 1 (by fitness) before migration
    best_idx = argmax(ind.fitness for ind in islands[1])
    best_fit = islands[1][best_idx].fitness

    migrate!(islands, CFG, MersenneTwister(5))

    # The sending island (1) should still have an individual with that fitness
    @test any(ind.fitness == best_fit for ind in islands[1])
end

@testset "migrate! :star — hub (island 1) receives migrants" begin
    cfg_star = GPConfig(
        population_size    = 20, n_islands = 4,
        n_trees_per_individual = 1,
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], migration_topology = :star,
        migration_rate = 0.2, migration_interval = 1,
        selection = :tournament, tournament_k = 3, lexicase_n_periods = 5,
        fitness_weights = FitnessWeights(0.6, 0.2, 0.0, 0.2),
        eval_subsample = 1.0, behavior_sample_size = 20,
    )
    pop     = _make_pop(20; seed = 10)
    islands = partition_population(pop, cfg_star)

    # Capture max fitness of non-hub islands before migration
    max_non_hub = maximum(maximum(ind.fitness for ind in islands[k])
                          for k in 2:4)

    migrate!(islands, cfg_star, MersenneTwister(11))

    # Hub should now contain at least one migrant from a non-hub island
    hub_fits = [ind.fitness for ind in islands[1]]
    @test any(f -> f ≈ max_non_hub || f >= max_non_hub - 1e-8, hub_fits)
end

@testset "migrate! :random — all island sizes preserved" begin
    cfg_rand = GPConfig(
        population_size    = 20, n_islands = 4,
        n_trees_per_individual = 1,
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], migration_topology = :random,
        migration_rate = 0.2, migration_interval = 1,
        selection = :tournament, tournament_k = 3, lexicase_n_periods = 5,
        fitness_weights = FitnessWeights(0.6, 0.2, 0.0, 0.2),
        eval_subsample = 1.0, behavior_sample_size = 20,
    )
    pop     = _make_pop(20; seed = 20)
    islands = partition_population(pop, cfg_rand)
    sizes   = [length(i) for i in islands]

    for _ in 1:10
        migrate!(islands, cfg_rand, MersenneTwister(rand(RNG, 1:10000)))
    end

    @test [length(i) for i in islands] == sizes
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "run_gp — returns a ParetoHoF" begin
    hof = run_gp(X_TEST, Y_RAW, CFG)
    @test hof isa ParetoHoF
end

@testset "run_gp — HoF is non-empty" begin
    hof = run_gp(X_TEST, Y_RAW, CFG)
    @test !isempty(hof.members)
end

@testset "run_gp — HoF never exceeds max_size" begin
    hof = run_gp(X_TEST, Y_RAW, CFG)
    @test length(hof.members) <= CFG.hof_size
end

@testset "run_gp — all HoF members are mutually non-dominated" begin
    hof = run_gp(X_TEST, Y_RAW, CFG)
    for i in eachindex(hof.members), j in eachindex(hof.members)
        i == j && continue
        @test !pareto_dominates(hof.members[i], hof.members[j])
    end
end

@testset "run_gp — HoF members have finite, evaluated fitness" begin
    hof = run_gp(X_TEST, Y_RAW, CFG)
    for m in hof.members
        @test isfinite(m.fitness)
        @test m.fitness != -Inf
    end
end

@testset "run_gp — deterministic with same seed" begin
    hof1 = run_gp(X_TEST, Y_RAW, CFG)
    hof2 = run_gp(X_TEST, Y_RAW, CFG)

    @test length(hof1.members) == length(hof2.members)
    pps1 = sort([m.pps_score for m in hof1.members])
    pps2 = sort([m.pps_score for m in hof2.members])
    @test pps1 ≈ pps2  atol=1e-8
end

@testset "run_gp — HoF members evaluate without error on unseen data" begin
    hof   = run_gp(X_TEST, Y_RAW, CFG)
    X_new = randn(MersenneTwister(99), 40, N_FEATURES)
    for m in hof.members
        out = combine_trees(m, X_new)
        @test out isa Vector{Float64}
        @test length(out) == 40
        @test !all(isnan, out)
    end
end

@testset "run_gp — verbose=false produces no output" begin
    # Just verifies it doesn't throw; output suppression is best-effort
    cfg_quiet = GPConfig(
        population_size = 20, n_islands = 2,
        n_trees_per_individual = 1,
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], n_generations = 2,
        elite_count = 1, selection = :tournament, tournament_k = 3,
        lexicase_n_periods = 5,
        fitness_weights = FitnessWeights(0.6, 0.2, 0.0, 0.2),
        eval_subsample = 1.0, behavior_sample_size = 10,
        migration_interval = 5, migration_rate = 0.2,
        const_opt_every_n_gens = 99, hof_size = 3,
        novelty_archive_size = 5, verbose = false,
    )
    @test_nowarn run_gp(X_TEST, Y_RAW, cfg_quiet)
end

@testset "run_gp — lexicase selection path works end-to-end" begin
    cfg_lex = GPConfig(
        population_size = 20, n_islands = 2,
        n_trees_per_individual = 1,
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], n_generations = 3,
        elite_count = 2, selection = :epsilon_lexicase,
        lexicase_epsilon = 0.1, lexicase_n_periods = 5,
        tournament_k = 3,
        fitness_weights = FitnessWeights(0.6, 0.2, 0.0, 0.2),
        eval_subsample = 1.0, behavior_sample_size = 10,
        migration_interval = 5, migration_rate = 0.2,
        const_opt_every_n_gens = 99, hof_size = 3,
        novelty_archive_size = 5, verbose = false,
    )
    hof = run_gp(X_TEST, Y_RAW, cfg_lex)
    @test hof isa ParetoHoF
    @test !isempty(hof.members)
end

end  # @testset

println("\n✓ All Island.jl + Engine.jl tests passed.")