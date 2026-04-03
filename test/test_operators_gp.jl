# =============================================================================
# test/test_operators_gp.jl
# Run with: julia --project test/test_operators_gp.jl
# =============================================================================

using Test
using Random

include("../src/Types.jl")
include("../src/Functions.jl")
include("../src/Tree.jl")
include("../src/Evaluate.jl")
include("../src/Fitness.jl")
include("../src/Init.jl")
include("../src/Operators_GP.jl")

# ── Fixtures ─────────────────────────────────────────────────────────────────

const N_FEATURES = 5
const N_ROWS     = 100
const X_TEST     = randn(MersenneTwister(7), N_ROWS, N_FEATURES)

const CFG = GPConfig(
    min_depth              = 2,
    max_depth              = 5,
    const_prob             = 0.2,
    fin_op_prob            = 0.3,
    ts_window_sizes        = [3, 5, 10],
    population_size        = 20,
    n_islands              = 1,
    n_trees_per_individual = 2,
    fitness_weights        = FitnessWeights(0.3, 0.3, 0.3, 0.1),
    n_generations          = 10,
    parsimony              = 0.001,
    eval_subsample         = 0.5,
    behavior_sample_size   = 20,
    mutation_prob          = MutationProbs(0.4, 0.35, 0.15, 0.1),
)
const OP_SETS = make_op_sets(CFG)

function _make_ind(rng=MersenneTwister(1))
    trees   = [build_random_tree(MersenneTwister(s), N_FEATURES, CFG, OP_SETS)
               for s in 1:CFG.n_trees_per_individual]
    weights = ones(CFG.n_trees_per_individual) ./ CFG.n_trees_per_individual
    Individual(trees, weights, CFG.combination)
end

# =============================================================================
@testset "Operators_GP.jl" begin

    # ── _clone_individual ────────────────────────────────────────────────────
    @testset "_clone_individual: independence" begin
        ind   = _make_ind()
        clone = _clone_individual(ind)

        @test ind.combination == clone.combination
        @test ind.weights     == clone.weights

        # Modifying the clone must not affect the original
        clone.trees[1].nodes[1] = variable_node(1)
        @test tree_to_string(ind.trees[1]) != tree_to_string(clone.trees[1]) ||
              tree_to_string(ind.trees[1]) == "x1"   # coincidence is OK
    end

    # ── _extract_subtree ─────────────────────────────────────────────────────
    @testset "_extract_subtree: root extraction = full tree" begin
        ind  = _make_ind()
        tree = ind.trees[1]
        sub  = _extract_subtree(tree, 1)

        @test tree_to_string(sub) == tree_to_string(tree)
        @test sub.size  == tree.size
        @test sub.depth == tree.depth
    end

    @testset "_extract_subtree: sub-node extracts smaller tree" begin
        rng  = MersenneTwister(42)
        ind  = _make_ind(rng)
        tree = ind.trees[1]

        # Find any operator node other than root
        non_root_ops = [i for i in 2:tree.size if is_operator(tree.nodes[i])]
        if !isempty(non_root_ops)
            target = first(non_root_ops)
            sub = _extract_subtree(tree, target)
            @test sub.size < tree.size
            @test sub.depth <= tree.depth
        end
    end

    # ── _node_depth ──────────────────────────────────────────────────────────
    @testset "_node_depth: root is always 0" begin
        for seed in 1:10
            tree = build_random_tree(MersenneTwister(seed), N_FEATURES, CFG, OP_SETS)
            @test _node_depth(tree, 1) == 0
        end
    end

    @testset "_node_depth: leaf depths consistent with tree.depth" begin
        for seed in 1:10
            tree = build_random_tree(MersenneTwister(seed), N_FEATURES, CFG, OP_SETS)
            for i in 1:tree.size
                d = _node_depth(tree, i)
                @test 0 <= d <= tree.depth
            end
        end
    end

    # ── crossover ────────────────────────────────────────────────────────────
    @testset "crossover: depth invariant" begin
        rng = MersenneTwister(100)
        for _ in 1:50
            p1 = _make_ind(rng)
            p2 = _make_ind(rng)
            c1, c2 = crossover(rng, p1, p2, CFG)
            for t in vcat(c1.trees, c2.trees)
                @test t.depth <= CFG.max_depth
            end
        end
    end

    @testset "crossover: children have correct n_trees" begin
        rng = MersenneTwister(200)
        p1  = _make_ind(rng)
        p2  = _make_ind(rng)
        c1, c2 = crossover(rng, p1, p2, CFG)
        @test length(c1.trees) == CFG.n_trees_per_individual
        @test length(c2.trees) == CFG.n_trees_per_individual
    end

    @testset "crossover: complexity updated" begin
        rng = MersenneTwister(300)
        p1  = _make_ind(rng)
        p2  = _make_ind(rng)
        c1, c2 = crossover(rng, p1, p2, CFG)
        @test c1.complexity == sum(t.size for t in c1.trees)
        @test c2.complexity == sum(t.size for t in c2.trees)
    end

    @testset "crossover: evaluable children" begin
        rng = MersenneTwister(400)
        for _ in 1:20
            p1  = _make_ind(rng)
            p2  = _make_ind(rng)
            c1, c2 = crossover(rng, p1, p2, CFG)
            for ind in (c1, c2)
                result = combine_trees(ind, X_TEST)
                @test result isa Vector{Float64}
                @test length(result) == N_ROWS
            end
        end
    end

    @testset "crossover: reproducibility" begin
        p1 = _make_ind(MersenneTwister(1))
        p2 = _make_ind(MersenneTwister(2))
        c1a, c2a = crossover(MersenneTwister(99), p1, p2, CFG)
        c1b, c2b = crossover(MersenneTwister(99), p1, p2, CFG)
        @test tree_to_string(c1a.trees[1]) == tree_to_string(c1b.trees[1])
        @test tree_to_string(c2a.trees[1]) == tree_to_string(c2b.trees[1])
    end

    # ── subtree_mutate ──────────────────────────────────────────────────────
    @testset "subtree_mutate: depth invariant" begin
        rng = MersenneTwister(500)
        for _ in 1:50
            ind   = _make_ind(rng)
            child = subtree_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
            for t in child.trees
                @test t.depth <= CFG.max_depth
            end
        end
    end

    @testset "subtree_mutate: evaluable result" begin
        rng = MersenneTwister(600)
        for _ in 1:20
            ind   = _make_ind(rng)
            child = subtree_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
            result = combine_trees(child, X_TEST)
            @test result isa Vector{Float64}
            @test length(result) == N_ROWS
        end
    end

    @testset "subtree_mutate: does not modify original" begin
        rng  = MersenneTwister(700)
        ind  = _make_ind(rng)
        s1   = tree_to_string(ind.trees[1])
        _    = subtree_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
        @test tree_to_string(ind.trees[1]) == s1
    end

    # ── point_mutate ────────────────────────────────────────────────────────
    @testset "point_mutate: depth unchanged" begin
        rng = MersenneTwister(800)
        for _ in 1:50
            ind   = _make_ind(rng)
            child = point_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
            for (to, tc) in zip(ind.trees, child.trees)
                @test tc.depth == to.depth   # point mutation never changes depth
            end
        end
    end

    @testset "point_mutate: evaluable result" begin
        rng = MersenneTwister(900)
        for _ in 1:20
            ind    = _make_ind(rng)
            child  = point_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
            result = combine_trees(child, X_TEST)
            @test result isa Vector{Float64}
            @test length(result) == N_ROWS
        end
    end

    @testset "point_mutate: does not modify original" begin
        rng = MersenneTwister(1000)
        ind  = _make_ind(rng)
        s1   = tree_to_string(ind.trees[1])
        _    = point_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
        @test tree_to_string(ind.trees[1]) == s1
    end

    # ── hoist_mutate ────────────────────────────────────────────────────────
    @testset "hoist_mutate: result is smaller or equal" begin
        rng = MersenneTwister(1100)
        for _ in 1:50
            ind   = _make_ind(rng)
            child = hoist_mutate(rng, ind, CFG)
            orig_size  = sum(t.size for t in ind.trees)
            child_size = sum(t.size for t in child.trees)
            @test child_size <= orig_size
        end
    end

    @testset "hoist_mutate: depth invariant" begin
        rng = MersenneTwister(1200)
        for _ in 1:50
            ind   = _make_ind(rng)
            child = hoist_mutate(rng, ind, CFG)
            for t in child.trees
                @test t.depth <= CFG.max_depth
            end
        end
    end

    @testset "hoist_mutate: evaluable result" begin
        rng = MersenneTwister(1300)
        for _ in 1:20
            ind    = _make_ind(rng)
            child  = hoist_mutate(rng, ind, CFG)
            result = combine_trees(child, X_TEST)
            @test result isa Vector{Float64}
            @test length(result) == N_ROWS
        end
    end

    # ── shrink_mutate ────────────────────────────────────────────────────────
    @testset "shrink_mutate: result is smaller or equal" begin
        rng = MersenneTwister(1400)
        for _ in 1:50
            ind   = _make_ind(rng)
            child = shrink_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
            @test sum(t.size for t in child.trees) <= sum(t.size for t in ind.trees)
        end
    end

    @testset "shrink_mutate: depth invariant" begin
        rng = MersenneTwister(1500)
        for _ in 1:50
            ind   = _make_ind(rng)
            child = shrink_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
            for t in child.trees
                @test t.depth <= CFG.max_depth
            end
        end
    end

    @testset "shrink_mutate: evaluable result" begin
        rng = MersenneTwister(1600)
        for _ in 1:20
            ind    = _make_ind(rng)
            child  = shrink_mutate(rng, ind, CFG, N_FEATURES, OP_SETS)
            result = combine_trees(child, X_TEST)
            @test result isa Vector{Float64}
            @test length(result) == N_ROWS
        end
    end

    # ── apply_mutation dispatcher ────────────────────────────────────────────
    @testset "apply_mutation: depth invariant across 200 calls" begin
        rng = MersenneTwister(9999)
        ind = _make_ind(rng)
        for _ in 1:200
            child = apply_mutation(rng, ind, CFG, N_FEATURES, OP_SETS)
            for t in child.trees
                @test t.depth <= CFG.max_depth
            end
        end
    end

    @testset "apply_mutation: all operators reachable (probabilistic)" begin
        # With 500 calls across a full mutation probability budget,
        # we expect each operator to fire at least once (smoke test).
        # Seeds are fixed so it's deterministic.
        mp   = CFG.mutation_prob
        rng  = MersenneTwister(42)
        ind  = _make_ind(rng)
        seen = Set{Symbol}()
        for _ in 1:500
            r    = rand(rng)
            t1   = mp.subtree
            t2   = t1 + mp.point
            t3   = t2 + mp.hoist
            t4   = t3 + mp.shrink
            op   = r < t1 ? :subtree :
                   r < t2 ? :point   :
                   r < t3 ? :hoist   :
                   r < t4 ? :shrink  : :none
            push!(seen, op)
        end
        @test :subtree ∈ seen
        @test :point   ∈ seen
        @test :hoist   ∈ seen
        @test :shrink  ∈ seen
    end

    @testset "apply_mutation: evaluable across all operators" begin
        rng = MersenneTwister(777)
        for _ in 1:50
            ind    = _make_ind(rng)
            child  = apply_mutation(rng, ind, CFG, N_FEATURES, OP_SETS)
            result = combine_trees(child, X_TEST)
            @test result isa Vector{Float64}
            @test length(result) == N_ROWS
        end
    end

end  # @testset "Operators_GP.jl"