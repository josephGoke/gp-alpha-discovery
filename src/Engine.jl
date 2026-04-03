# =============================================================================
# Engine.jl — Main GP orchestration loop
#
# Entry point: run_gp(X, y_raw, config) → ParetoHoF
#
# Per-generation loop (per island):
#   1. evaluate_population!       — fitness (pps, rre), behavior fingerprints
#   2. compute_novelty_scores     — knn behavioral distance → ind.novelty_score
#   3. _apply_novelty_to_fitness! — add novelty bonus to ind.fitness in-place
#   4. update_archive!            — archive high-novelty individuals
#   5. update_hof!                — Pareto front (PPS, RRE, complexity)
#   6. optimize_population_constants! — elite constant tuning (every N gens)
#   7. build_error_matrix + select → crossover/mutation → next generation
#   8. migrate! (every migration_interval gens)
#   9. log generation stats (if config.verbose)
#
# Public API
# ──────────────────────────────────────────────────────────────────────────
#   run_gp(X, y_raw, config) → ParetoHoF
# =============================================================================

using Random
using Statistics


# =============================================================================
# 1. Novelty fitness adjustment
# =============================================================================

"""
    _apply_novelty_to_fitness!(island, novelty_scores, config)

Add the novelty bonus to each individual's fitness in-place.

  ind.fitness += config.fitness_weights.novelty * novelty_scores[i]

Called after `compute_novelty_scores` returns scores and after
`evaluate_fitness!` has computed base fitness (without novelty).
No tree re-evaluation needed.
"""
function _apply_novelty_to_fitness!(island::Vector{Individual},
                                     novelty_scores::Vector{Float64},
                                     config::GPConfig)
    nw = config.fitness_weights.novelty
    nw == 0.0 && return nothing
    for (i, ind) in enumerate(island)
        ind.fitness += nw * novelty_scores[i]
    end
    nothing
end


# =============================================================================
# 2. Next-generation builder (selection → crossover/mutation)
# =============================================================================

"""
    _build_next_generation(island, errors, config, n_features, op_sets, rng)
                → Vector{Individual}

Produce the next generation for one island:

1. Elitism: copy the top `elite_count` individuals unchanged.
2. Fill the rest via crossover (prob = crossover_prob) or mutation.
   Parents are drawn via `select` (tournament or epsilon-lexicase).

The new generation is exactly `length(island)` individuals.
"""
function _build_next_generation(island::Vector{Individual},
                                 errors::Matrix{Float64},
                                 config::GPConfig,
                                 n_features::Int,
                                 op_sets::NamedTuple,
                                 rng::AbstractRNG)::Vector{Individual}
    n         = length(island)
    n_elite   = min(config.elite_count, n)
    next_gen  = Vector{Individual}(undef, n)

    # Elites — sorted descending by fitness, copied directly
    elite_order = sortperm(island; by = ind -> ind.fitness, rev = true)
    for k in 1:n_elite
        next_gen[k] = island[elite_order[k]]
    end

    # Fill remaining slots
    cursor = n_elite + 1
    while cursor <= n
        if rand(rng) < config.crossover_prob
            p1 = select(island, errors, config, rng)
            p2 = select(island, errors, config, rng)
            c1, c2 = crossover(rng, p1, p2, config)
            next_gen[cursor] = c1
            cursor += 1
            if cursor <= n
                next_gen[cursor] = c2
                cursor += 1
            end
        else
            parent = select(island, errors, config, rng)
            next_gen[cursor] = apply_mutation(rng, parent, config,
                                               n_features, op_sets)
            cursor += 1
        end
    end

    return next_gen
end


# =============================================================================
# 3. Per-generation logging
# =============================================================================

"""
    _log_generation(gen, islands, hof, archive, config)

Print a one-line summary for generation `gen`.
"""
function _log_generation(gen::Int,
                          islands::Vector{Vector{Individual}},
                          hof::ParetoHoF,
                          archive::NoveltyArchive,
                          island_novelty_scores::Vector{Vector{Float64}},
                          config::GPConfig)
    all_inds  = vcat(islands...)
    best_fit  = maximum(ind.fitness   for ind in all_inds)
    
    mean_fit  = mean(ind.fitness      for ind in all_inds)
    best_pps  = maximum(ind.pps_score for ind in all_inds)
    mean_nov  = mean(mean(s) for s in island_novelty_scores)
    hof_size  = length(hof.members)
    arch_size = length(archive.behaviors)

    @info "Gen $gen | " *
          "best_fit=$(round(best_fit, digits=4)) | " *
          "mean_fit=$(round(mean_fit, digits=4)) | " *
          "best_pps=$(round(best_pps, digits=4)) | " *
          "novelty=$(round(mean_nov, digits=4)) | " *
          "HoF=$hof_size | archive=$arch_size"
end


# =============================================================================
# 4. Main loop
# =============================================================================

"""
    run_gp(X, y_raw, config) → ParetoHoF

Run the full GP search and return the Pareto Hall of Fame.

Arguments
─────────
- `X`      : training feature matrix (n_rows × n_features), temporally ordered
- `y_raw`  : target return vector, length n_rows
- `config` : `GPConfig` — all hyperparameters

Returns
───────
A `ParetoHoF` whose `members` are the non-dominated individuals on the
(PPS ↑, RRE ↑, complexity ↓) Pareto front discovered during the run.

Usage
─────
```julia
hof = run_gp(X_train, y_train, GPConfig(n_generations=100, population_size=500))
best = hof.members[argmax(m.pps_score for m in hof.members)]
signal = combine_trees(best, X_test)
```
"""
function run_gp(X_train::Matrix{Float64},
                y_train_raw::Vector{Float64},
                config::GPConfig)::ParetoHoF

    validate_config(config)

    n_rows, n_features = size(X_train)
    rng = MersenneTwister(config.seed)

    # ── Startup ───────────────────────────────────────────────────────────────
    op_sets      = make_op_sets(config)
    population   = init_population(rng, n_features, config, op_sets)
    islands      = partition_population(population, config)
    behavior_rows = init_behavior_rows(rng, n_rows, config)
    archive      = NoveltyArchive(config.novelty_archive_size)
    hof          = ParetoHoF(config.hof_size)
    y_ranked     = prerank_y(y_train_raw)

    # One RNG per island — deterministic, independent of number of threads
    island_rngs = [MersenneTwister(config.seed + k) for k in 1:config.n_islands]

    config.verbose && @info "GP started: $(config.population_size) individuals, " *
                            "$(config.n_islands) islands, " *
                            "$(config.n_generations) generations"

    # ── Generation loop ───────────────────────────────────────────────────────
    for gen in 1:config.n_generations

        # ── Per-island evolution ─────────────────────────────────────────────
        # Collect per-island novelty scores locally — Individual has no
        # novelty_score field; scores are transient and only used this generation.
        island_novelty_scores = Vector{Vector{Float64}}(undef, length(islands))

        for (k, island) in enumerate(islands)
            irng = island_rngs[k]

            # Step 1 — evaluate fitness + behavior fingerprints
            evaluate_population!(island, X_train, y_train_raw, behavior_rows, config, irng, gen)

            # Step 2 — novelty scores (population ∪ archive)
            nov_scores = compute_novelty_scores(island, archive, config)
            island_novelty_scores[k] = nov_scores

            # Step 3 — add novelty bonus to fitness (no tree re-eval)
            _apply_novelty_to_fitness!(island, nov_scores, config)
        end

        # ── Global diversity / HoF updates (across all islands) ──────────────
        all_inds = vcat(islands...)

        # Step 4 — archive high-novelty individuals
        update_archive!(archive, all_inds, config)

        # Step 5 — Pareto HoF
        update_hof!(hof, all_inds)

        # ── Per-island: constant optimisation + next generation ───────────────
        for (k, island) in enumerate(islands)
            irng = island_rngs[k]

            # Step 6 — constant tuning for elites (every N gens)
            optimize_population_constants!(island, X_train, y_train_raw, y_ranked, config, gen)

            # Step 7 — build error matrix for lexicase selection
            errors = build_error_matrix(island, X_train, y_train_raw, config)

            # Step 8 — selection → crossover/mutation → next generation
            islands[k] = _build_next_generation(island, errors, config,
                                                 n_features, op_sets, irng)
        end

        # ── Step 9 — migration (every migration_interval generations) ─────────
        if gen % config.migration_interval == 0
            migrate!(islands, config, rng)
        end

        # ── Logging ───────────────────────────────────────────────────────────
        config.verbose && _log_generation(gen, islands, hof, archive,
                                           island_novelty_scores, config)
    end

    config.verbose && @info "GP complete. HoF: $(length(hof.members)) members."
    return hof
end