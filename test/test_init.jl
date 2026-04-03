# =============================================================================
# test/test_init.jl
# Run with: julia --project test/test_init.jl
# =============================================================================

using Test
using Random
using Statistics

include("../src/Types.jl")
include("../src/Functions.jl")
include("../src/Tree.jl")
include("../src/Evaluate.jl")
include("../src/Fitness.jl")
include("../src/Init.jl")

# ── Shared fixtures ───────────────────────────────────────────────────────────

const N_FEATURES = 5
const N_ROWS     = 100
const X_TEST     = randn(MersenneTwister(7), N_ROWS, N_FEATURES)

const CFG = GPConfig(
    min_depth            = 2,
    max_depth            = 5,
    const_prob           = 0.2,
    fin_op_prob          = 0.3,
    ts_window_sizes      = [3, 5, 10],
    population_size      = 100,
    n_islands            = 2,
    fitness_weights      = FitnessWeights(0.3, 0.3, 0.3, 0.1),
    n_generations        = 50,
    parsimony            = 0.001,
    eval_subsample       = 0.5,
    behavior_sample_size = 20,
)

const OP_SETS = make_op_sets(CFG)

# =============================================================================
@testset "Init.jl — Ramped Half-and-Half" begin

    # ── 1. Basic size & shape ────────────────────────────────────────────────
    @testset "population size" begin
        rng = MersenneTwister(42)
        pop = init_population(rng, N_FEATURES, CFG, OP_SETS)

        @test length(pop) == CFG.population_size
        for ind in pop
            @test length(ind.trees)   == CFG.n_trees_per_individual
            @test length(ind.weights) == CFG.n_trees_per_individual
        end
    end

    # ── 2. Weight sanity ────────────────────────────────────────────────────
    @testset "equal weights sum to 1" begin
        rng = MersenneTwister(1)
        pop = init_population(rng, N_FEATURES, CFG, OP_SETS)
        for ind in pop
            @test sum(ind.weights) ≈ 1.0 atol=1e-10
            @test all(w ≈ (1.0 / CFG.n_trees_per_individual) for w in ind.weights)
        end
    end

    # ── 3. Combination field propagated from config ──────────────────────────
    @testset "combination field" begin
        rng = MersenneTwister(2)
        pop = init_population(rng, N_FEATURES, CFG, OP_SETS)
        for ind in pop
            @test ind.combination == CFG.combination
        end
    end

    # ── 4. Depth diversity ───────────────────────────────────────────────────
    @testset "depth diversity across population" begin
        rng = MersenneTwister(3)
        pop = init_population(rng, N_FEATURES, CFG, OP_SETS)

        # Collect max depth of each tree across the population
        tree_depths = Int[]
        for ind in pop, tree in ind.trees
            push!(tree_depths, tree.depth)
        end

        min_obs = minimum(tree_depths)
        max_obs = maximum(tree_depths)

        # We expect depth coverage from min_depth to max_depth (or close)
        @test min_obs >= CFG.min_depth
        @test max_obs <= CFG.max_depth
        # At least 3 distinct depth levels should appear in a pop of 100
        @test length(unique(tree_depths)) >= 3
    end

    # ── 5. No degenerate trees (every tree must be non-empty) ────────────────
    @testset "non-empty trees" begin
        rng = MersenneTwister(4)
        pop = init_population(rng, N_FEATURES, CFG, OP_SETS)
        for ind in pop, tree in ind.trees
            @test length(tree.nodes) >= 1
        end
    end

    # ── 6. All trees are evaluable ──────────────────────────────────────────
    @testset "trees evaluate without error" begin
        rng = MersenneTwister(5)
        pop = init_population(rng, N_FEATURES, CFG, OP_SETS)
        for ind in pop
            result = combine_trees(ind, X_TEST)
            @test result isa Vector{Float64}
            @test length(result) == N_ROWS
            # No NaN/Inf in result (trees may produce small values but not all NaN)
            @test !all(isnan, result)
        end
    end

    # ── 7. Reproducibility ──────────────────────────────────────────────────
    @testset "same seed → same population" begin
        pop_a = init_population(MersenneTwister(99), N_FEATURES, CFG, OP_SETS)
        pop_b = init_population(MersenneTwister(99), N_FEATURES, CFG, OP_SETS)

        for (a, b) in zip(pop_a, pop_b)
            @test a.combination == b.combination
            @test a.weights     == b.weights
            @test length(a.trees) == length(b.trees)
            for (ta, tb) in zip(a.trees, b.trees)
                @test tree_to_string(ta) == tree_to_string(tb)
            end
        end
    end

    # ── 8. Different seeds → different populations ──────────────────────────
    @testset "different seeds → different populations" begin
        pop_a = init_population(MersenneTwister(1), N_FEATURES, CFG, OP_SETS)
        pop_b = init_population(MersenneTwister(2), N_FEATURES, CFG, OP_SETS)

        # Very unlikely all tree strings match under different seeds
        n_same = sum(tree_to_string(a.trees[1]) == tree_to_string(b.trees[1])
                     for (a, b) in zip(pop_a, pop_b))
        @test n_same < CFG.population_size  # at least some differ
    end

    # ── 9. Fitness fields initialise to defaults ─────────────────────────────
    @testset "individual fitness fields initialised" begin
        rng = MersenneTwister(6)
        pop = init_population(rng, N_FEATURES, CFG, OP_SETS)
        for ind in pop
            @test !isnan(ind.fitness)   # default is -Inf, not NaN
            @test ind.age      == 0
            @test ind.dominated == false
        end
    end

    # ── 10. Remainder allocation — pop size not divisible by n_depths ────────
    @testset "remainder allocation" begin
        # Create a config where population_size is not divisible by n_depths
        n_depths = CFG.max_depth - CFG.min_depth + 1   # =4 for depths 2..5
        pop_size = n_depths * 10 + 3                   # =43, remainder=3

        cfg_odd = GPConfig(
            min_depth            = CFG.min_depth,
            max_depth            = CFG.max_depth,
            const_prob           = CFG.const_prob,
            fin_op_prob          = CFG.fin_op_prob,
            ts_window_sizes      = CFG.ts_window_sizes,
            population_size      = pop_size,
            n_islands            = 1,
            fitness_weights      = CFG.fitness_weights,
            n_generations        = CFG.n_generations,
            parsimony            = CFG.parsimony,
            eval_subsample       = CFG.eval_subsample,
            behavior_sample_size = CFG.behavior_sample_size,
        )
        op_odd = make_op_sets(cfg_odd)
        rng    = MersenneTwister(77)
        pop    = init_population(rng, N_FEATURES, cfg_odd, op_odd)

        @test length(pop) == pop_size
    end

end  # @testset "Init.jl — Ramped Half-and-Half"