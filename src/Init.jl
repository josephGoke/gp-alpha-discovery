# =============================================================================
# Init.jl — Ramped Half-and-Half population initialisation
#
# Public API
# ─────────────────────────────────────────────────────────────────────────────
#   init_population(rng, n_features, config, op_sets) → Vector{Individual}
#
# Overview
# ─────────────────────────────────────────────────────────────────────────────
# 
# Uses Ramped Half-and-Half (Koza 1992):
#   - Population split evenly across depths min_depth:max_depth
#   - At each depth level: 50% built with :full, 50% with :grow
#   - Each individual contains n_trees_per_individual trees,
#     each built independently with its own rng state
#
# This guarantees structural diversity at generation 0, giving selection
# genuine signal to work with from the very first generation.
#
# Depth  | Fraction of pop   | Split
# ───────────────────────────────────────
# d = 2  | 1/(max-min+1)     | 50% full / 50% grow
# d = 3  |        …          |        …
# d = max|        …          |        …
#
# =============================================================================

using Random


"""
    init_individual(rng, n_features, config, op_sets, target_depth) → Individual

Build a single `Individual` with `config.n_trees_per_individual` trees.
Each tree is built with alternating :full / :grow methods at `target_depth`,
using a fresh split of the rng state per tree for reproducibility.
"""
function init_individual(rng::AbstractRNG, n_features::Int, config::GPConfig,
                          op_sets::NamedTuple, target_depth::Int=config.max_depth)::Individual
    n_trees = config.n_trees_per_individual
    trees   = Vector{Tree}(undef, n_trees)

    for i in 1:n_trees
        # Alternate full / grow within the same individual for variety
        method = isodd(i) ? :full : :grow
        trees[i] = build_random_tree(rng, n_features, config, op_sets;
                                     method    = method,
                                     max_depth = target_depth)
    end

    weights = ones(Float64, n_trees) ./ n_trees
    return Individual(trees, weights, config.combination)
end




"""
    init_population(rng, n_features, config, op_sets) → Vector{Individual}

Build a full population of `config.population_size` individuals using
Ramped Half-and-Half initialisation.

Steps
─────
1. Compute the depth range: `min_depth` to `max_depth` (inclusive).
2. Divide `population_size` evenly across depth levels.
   Any remainder is distributed one-per-level from the shallowest depth up.
3. For each depth level, build the allocated quota of individuals via
   `init_individual`, alternating :full / :grow at the level boundary.

The returned vector is in randomised order (shallow-to-deep then shuffled)
so callers can safely slice off islands without depth bias.
"""
function init_population(rng::AbstractRNG, n_features::Int, config::GPConfig,
                          op_sets::NamedTuple)::Vector{Individual}

    pop_size   = config.population_size
    depths      = config.min_depth : config.max_depth
    n_depths   = length(depths)          # number of depth levels

    # How many individuals per depth level — distribute remainder evenly
    base, rem = divrem(pop_size, n_depths)
    band_counts = fill(base, n_depths)
    for i in n_depths:-1:(n_depths - rem + 1)
        band_counts[i] += 1
    end

   

    population = Vector{Individual}(undef, pop_size)
    cursor     = 1

    for (level_idx, depth) in enumerate(depths)
        count = band_counts[level_idx]
    
        for _ in 1:count
            population[cursor] = init_individual(rng, n_features, config,
                                                  op_sets, depth)
            cursor += 1
        end
    end

    # Shuffle so island partitioning doesn't get all-shallow or all-deep chunks
    shuffle!(rng, population)
    return population
end




# =============================================================================
# 3. Behavior row sampling  (called once at engine startup)
# =============================================================================

"""
    init_behavior_rows(rng, n_rows, config) → Vector{Int}

Sample `config.behavior_sample_size` row indices without replacement from
`1:n_rows`.  These rows are fixed for the entire run — every individual's
behavioral fingerprint is evaluated on the same subset, making novelty
distances directly comparable across generations and the archive.

Errors if `behavior_sample_size > n_rows`.
"""
function init_behavior_rows(rng::AbstractRNG, n_rows::Int,
                             config::GPConfig)::Vector{Int}
    config.behavior_sample_size <= n_rows ||
        error("init_behavior_rows: behavior_sample_size " *
              "($(config.behavior_sample_size)) > n_rows ($n_rows)")
    return randperm(rng, n_rows)[1:config.behavior_sample_size]
end