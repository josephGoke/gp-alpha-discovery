# =============================================================================
# test/test_diversity.jl
# Run with: julia --project test/test_diversity.jl
# =============================================================================

using Test
using Random
using Statistics

include("../src/Genetic.jl")
using .Genetic

const RNG        = MersenneTwister(42)
const N_FEATURES = 5
const N_ROWS     = 120
const BEH_SIZE   = 20
const X_TEST     = randn(RNG, N_ROWS, N_FEATURES)
const Y_RAW      = X_TEST[:, 1] .+ 0.2 .* randn(RNG, N_ROWS)
const Y_RANKED   = prerank_y(Y_RAW)
const BEH_ROWS   = collect(1:BEH_SIZE)

const CFG = GPConfig(
    min_depth              = 2,
    max_depth              = 4,
    const_prob             = 0.2,
    fin_op_prob            = 0.0,
    ts_window_sizes        = [3],
    population_size        = 20,
    n_islands              = 1,
    n_trees_per_individual = 1,
    n_generations          = 10,
    novelty_k              = 3,
    novelty_archive_size   = 10,
    behavior_sample_size   = BEH_SIZE,
    fitness_weights        = FitnessWeights(0.3, 0.3, 0.3, 0.1),
    parsimony              = 0.001,
    eval_subsample         = 1.0,
    hof_size               = 5,
)
const OP_SETS = make_op_sets(CFG)

_rand_beh(len::Int = BEH_SIZE; rng = RNG) = rand(rng, Float32, len)

function _ind(; pps::Float64 = 0.5, rre::Float64 = 0.5, complexity::Int = 10)
    tree = Tree([variable_node(1)], Float64[], 1, 0)
    ind  = Individual([tree], [1.0], :weighted_sum)
    ind.pps_score  = pps
    ind.rre_score  = rre
    ind.complexity = complexity
    ind.fitness    = pps
    ind.behavior   = _rand_beh()
    return ind
end

function _make_population(n::Int)
    pop = Vector{Individual}(undef, n)
    for i in 1:n
        tree = build_random_tree(MersenneTwister(i), N_FEATURES, CFG, OP_SETS)
        ind  = Individual([tree], [1.0], :weighted_sum)
        update_complexity!(ind)
        evaluate_fitness!(ind, X_TEST, Y_RAW, Y_RANKED, CFG)
        update_behavior!(ind, X_TEST, BEH_ROWS)
        pop[i] = ind
    end
    return pop
end

@testset "Diversity.jl" begin

@testset "behavioral_distance — basic geometry" begin
    b1 = Float32[1.0, 0.0];  b2 = Float32[0.0, 1.0];  b3 = Float32[1.0, 0.0]
    @test behavioral_distance(b1, b2) ≈ sqrt(2.0)  atol=1e-6
    @test behavioral_distance(b1, b3) ≈ 0.0        atol=1e-10
    @test behavioral_distance(b1, b2) == behavioral_distance(b2, b1)
end

@testset "behavioral_distance — length mismatch throws" begin
    @test_throws ErrorException behavioral_distance(Float32[1.0], Float32[1.0, 2.0])
end

@testset "knn_mean_distance — empty pool returns 0.0" begin
    @test knn_mean_distance(Float32[1.0, 2.0], Vector{Float32}[], 3) == 0.0
end

@testset "knn_mean_distance — k larger than pool uses full pool" begin
    q = Float32[0.0, 0.0]
    others = [Float32[1.0, 0.0], Float32[0.0, 2.0]]
    expected = mean([behavioral_distance(q, o) for o in others])
    @test knn_mean_distance(q, others, 10) ≈ expected  atol=1e-6
end

@testset "knn_mean_distance — k=1 returns nearest neighbor" begin
    q = Float32[0.0]
    @test knn_mean_distance(q, [Float32[0.1], Float32[100.0]], 1) ≈
          behavioral_distance(q, Float32[0.1])  atol=1e-6
end

@testset "compute_novelty_scores — shape, finite, non-negative" begin
    pop    = _make_population(8)
    archive = NoveltyArchive(CFG.novelty_archive_size)
    scores = compute_novelty_scores(pop, archive, CFG)
    @test length(scores) == 8
    @test all(isfinite, scores)
    @test all(scores .>= 0.0)
end

@testset "compute_novelty_scores — distinct behaviors give non-zero scores" begin
    pop = [_ind() for _ in 1:6]
    for i in eachindex(pop)
        pop[i].behavior = Float32.([(i-1)*10.0 + j for j in 1:BEH_SIZE])
    end
    scores = compute_novelty_scores(pop, NoveltyArchive(10), CFG)
    @test all(scores .> 0.0)
end

@testset "compute_novelty_scores — archive expands novelty signal" begin
    pop = [_ind() for _ in 1:4]
    for i in eachindex(pop)
        pop[i].behavior = Float32.(fill(Float32(i), BEH_SIZE))
    end
    arch_empty = NoveltyArchive(20)
    arch_far   = NoveltyArchive(20)
    push!(arch_far.behaviors, Float32.(fill(100.0f0, BEH_SIZE)))
    s_empty = compute_novelty_scores(pop, arch_empty, CFG)
    s_far   = compute_novelty_scores(pop, arch_far,   CFG)
    @test sum(s_far) >= sum(s_empty)
end

@testset "update_archive! — admits novel individuals" begin
    pop = [_ind() for _ in 1:5]
    for i in eachindex(pop)
        pop[i].behavior = Float32.(fill(Float32(i * 10), BEH_SIZE))
    end
    archive = NoveltyArchive(20)
    update_archive!(archive, pop, CFG)
    @test length(archive.behaviors) >= 1
end

@testset "update_archive! — never exceeds max_size" begin
    pop = [_ind() for _ in 1:50]
    for i in 1:50
        pop[i].behavior = _rand_beh(BEH_SIZE; rng = MersenneTwister(i))
    end
    archive = NoveltyArchive(10)
    update_archive!(archive, pop, CFG)
    @test length(archive.behaviors) <= 10
end

@testset "update_archive! — skips empty behavior vectors" begin
    pop = [_ind() for _ in 1:5]
    for ind in pop; ind.behavior = Float32[]; end
    archive = NoveltyArchive(10)
    update_archive!(archive, pop, CFG)
    @test isempty(archive.behaviors)
end

@testset "update_archive! — eviction keeps novel entry, drops least novel" begin
    archive = NoveltyArchive(3)
    push!(archive.behaviors, Float32[0.1, 0.0])
    push!(archive.behaviors, Float32[0.2, 0.0])
    push!(archive.behaviors, Float32[0.3, 0.0])

    novel = _ind()
    novel.behavior = Float32[100.0, 100.0]

    cfg_k1 = GPConfig(
        min_depth = 2, max_depth = 4, const_prob = 0.2, fin_op_prob = 0.0,
        ts_window_sizes = [3], population_size = 5, n_islands = 1,
        novelty_k = 1, novelty_archive_size = 3, behavior_sample_size = 2,
    )
    update_archive!(archive, [novel], cfg_k1)

    @test length(archive.behaviors) == 3
    @test any(b -> b == Float32[100.0, 100.0], archive.behaviors)
end

@testset "pareto_dominates — correct 3-objective logic" begin
    a = _ind(pps=0.8, rre=0.7, complexity=5)
    b = _ind(pps=0.5, rre=0.5, complexity=10)
    @test  pareto_dominates(a, b)
    @test !pareto_dominates(b, a)

    # Equal — neither dominates
    c = _ind(pps=0.5, rre=0.5, complexity=10)
    @test !pareto_dominates(b, c)
    @test !pareto_dominates(c, b)

    # Trade-off in PPS vs RRE — neither dominates
    d = _ind(pps=0.9, rre=0.3, complexity=10)
    e = _ind(pps=0.3, rre=0.9, complexity=10)
    @test !pareto_dominates(d, e)
    @test !pareto_dominates(e, d)

    # Complexity advantage only
    h = _ind(pps=0.5, rre=0.5, complexity=5)
    ii = _ind(pps=0.5, rre=0.5, complexity=10)
    @test  pareto_dominates(h, ii)
    @test !pareto_dominates(ii, h)
end

@testset "update_hof! — admits non-dominated, excludes dominated" begin
    hof = ParetoHoF(10)
    pop = [
        _ind(pps=0.8, rre=0.7, complexity=5),
        _ind(pps=0.5, rre=0.9, complexity=8),
        _ind(pps=0.3, rre=0.3, complexity=20),  # dominated
    ]
    update_hof!(hof, pop)
    @test length(hof.members) == 2
end

@testset "update_hof! — evicts dominated members when better arrives" begin
    hof  = ParetoHoF(10)
    weak = _ind(pps=0.4, rre=0.4, complexity=15)
    update_hof!(hof, [weak])
    @test length(hof.members) == 1

    strong = _ind(pps=0.9, rre=0.9, complexity=5)
    update_hof!(hof, [strong])
    @test length(hof.members) == 1
    @test hof.members[1] === strong
end

@testset "update_hof! — skips unevaluated (fitness = -Inf)" begin
    hof   = ParetoHoF(10)
    uneval = Individual([Tree([variable_node(1)], Float64[], 1, 0)], [1.0], :weighted_sum)
    @test uneval.fitness == -Inf
    update_hof!(hof, [uneval])
    @test isempty(hof.members)
end

@testset "update_hof! — never exceeds max_size" begin
    hof = ParetoHoF(3)
    pop = [_ind(pps=0.1*i, rre=1.0-0.1*i, complexity=i) for i in 1:10]
    update_hof!(hof, pop)
    @test length(hof.members) <= 3
end

@testset "update_hof! — all members mutually non-dominated" begin
    hof = ParetoHoF(10)
    pop = [_ind(pps=0.1*i, rre=1.0-0.1*i, complexity=5) for i in 1:8]
    update_hof!(hof, pop)
    for i in eachindex(hof.members), j in eachindex(hof.members)
        i == j && continue
        @test !pareto_dominates(hof.members[i], hof.members[j])
    end
end

@testset "update_hof! — idempotent re-insertion" begin
    hof = ParetoHoF(10)
    pop = [_ind(pps=0.1*i, rre=1.0-0.1*i, complexity=5) for i in 1:5]
    update_hof!(hof, pop)
    n1 = length(hof.members)
    update_hof!(hof, pop)
    @test length(hof.members) == n1
end

@testset "hof_summary — correct output format" begin
    hof = ParetoHoF(5)
    @test hof_summary(hof) == "HoF: empty"
    update_hof!(hof, [_ind(pps=0.7, rre=0.6, complexity=8)])
    s = hof_summary(hof)
    @test occursin("HoF:", s)
    @test occursin("best_pps", s)
    @test occursin("best_rre", s)
    @test occursin("complexity", s)
end

@testset "pps_score and rre_score cached by evaluate_fitness!" begin
    pop = _make_population(5)
    for ind in pop
        @test isfinite(ind.pps_score)
        @test isfinite(ind.rre_score)
        @test 0.0 <= ind.rre_score <= 1.0
    end
end

@testset "end-to-end: novelty + HoF over small population" begin
    pop     = _make_population(10)
    archive = NoveltyArchive(CFG.novelty_archive_size)
    hof     = ParetoHoF(CFG.hof_size)

    scores = compute_novelty_scores(pop, archive, CFG)
    @test length(scores) == 10
    @test all(scores .>= 0.0)

    update_archive!(archive, pop, CFG)
    @test length(archive.behaviors) <= CFG.novelty_archive_size

    update_hof!(hof, pop)
    @test 1 <= length(hof.members) <= CFG.hof_size

    for i in eachindex(hof.members), j in eachindex(hof.members)
        i == j && continue
        @test !pareto_dominates(hof.members[i], hof.members[j])
    end

    println(hof_summary(hof))
end

end  # @testset

println("\n✓ All Diversity.jl tests passed.")