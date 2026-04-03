# =============================================================================
# Island.jl — Island population partitioning and migration
#
# The island model splits the population into `n_islands` sub-populations.
# Each island evolves independently for `migration_interval` generations,
# then the best `ceil(migration_rate * island_size)` individuals from each
# island migrate to a target island determined by the topology.
#
# Migration is synchronous (all islands exchange in the same step) and
# single-process (no Distributed.jl). Distributed wiring is a thin wrapper
# around these same functions and can be added later without changing the logic.
#
# Topology semantics
# ──────────────────────────────────────────────────────────────────────────
#   :ring    Each island i sends migrants to island (i mod n) + 1.
#            Direction: 0→1→2→…→(n-1)→0. Promotes gradual spread.
#
#   :star    All islands send migrants to island 0 (the hub).
#            Island 0 accumulates best individuals from all others.
#            Non-hub islands do NOT receive migrants.
#
#   :random  Each island sends migrants to a uniformly random target
#            (excluding itself). Different target each migration event.
#
# Migration policy
# ──────────────────────────────────────────────────────────────────────────
#   Emigrants  : top `n_migrants` individuals by fitness (sorted descending)
#   Replacement: the `n_migrants` worst individuals on the receiving island
#                are replaced. This gives migrants an immediate foothold.
#   Deep copy  : migrants are copied — the sending island keeps its originals.
#
# Public API
# ──────────────────────────────────────────────────────────────────────────
#   partition_population(population, config)          → Vector{Vector{Individual}}
#   merge_islands(islands)                            → Vector{Individual}
#   migration_targets(n_islands, topology, rng)       → Vector{Int}
#   migrate!(islands, config, rng)                    → Nothing
# =============================================================================

using Random


# =============================================================================
# 1. Partition / merge
# =============================================================================

"""
    partition_population(population, config) → Vector{Vector{Individual}}

Split `population` into `config.n_islands` contiguous sub-populations of
equal size (`population_size ÷ n_islands` each).

`validate_config` enforces `population_size % n_islands == 0`, so the
split is always exact.
"""
function partition_population(population::Vector{Individual},
                               config::GPConfig)::Vector{Vector{Individual}}
    n          = length(population)
    n_islands  = config.n_islands
    island_size = div(n, n_islands)

    return [population[(k-1)*island_size + 1 : k*island_size]
            for k in 1:n_islands]
end


"""
    merge_islands(islands) → Vector{Individual}

Concatenate all island sub-populations back into a single flat vector.
Order: island 1 first, island n_islands last.
"""
function merge_islands(islands::Vector{Vector{Individual}})::Vector{Individual}
    return vcat(islands...)
end


# =============================================================================
# 2. Topology — migration target computation
# =============================================================================

"""
    migration_targets(n_islands, topology, rng) → Vector{Int}

Return a length-`n_islands` vector where `targets[i]` is the index of the
island that island `i` sends its emigrants to.

  :ring    targets[i] = i % n_islands + 1   (0-indexed mod, 1-indexed result)
  :star    targets[i] = 1  for all i ≠ 1;   targets[1] = 1 (hub receives all,
                                             self-send is a no-op)
  :random  targets[i] = uniformly random j ∈ 1:n_islands, j ≠ i
"""
function migration_targets(n_islands::Int,
                            topology::Symbol,
                            rng::AbstractRNG)::Vector{Int}
    n_islands == 1 && return [1]

    if topology == :ring
        return [i % n_islands + 1 for i in 1:n_islands]

    elseif topology == :star
        # Every island sends to hub (island 1); hub's own send is a no-op
        return ones(Int, n_islands)

    elseif topology == :random
        targets = Vector{Int}(undef, n_islands)
        for i in 1:n_islands
            options = [j for j in 1:n_islands if j != i]
            targets[i] = rand(rng, options)
        end
        return targets

    else
        error("migration_targets: unknown topology ':$topology'. " *
              "Valid: :ring, :star, :random")
    end
end


# =============================================================================
# 3. Core migration step
# =============================================================================

"""
    migrate!(islands, config, rng) → Nothing

Exchange individuals between islands according to `config.migration_topology`.

Per sending island:
1. Sort by fitness descending.
2. Take the top `n_migrants = max(1, ceil(Int, island_size * migration_rate))`
   individuals as emigrants.
3. Deep-copy them to the target island.
4. On the target island, replace the `n_migrants` worst individuals
   (lowest fitness) with the incoming migrants.

Self-sends (target == source) are silently skipped — nothing changes.
The sending island always keeps its own originals.
"""
function migrate!(islands::Vector{Vector{Individual}},
                  config::GPConfig,
                  rng::AbstractRNG)
    n_islands   = length(islands)
    n_islands <= 1 && return nothing

    island_size = length(islands[1])
    n_migrants  = max(1, ceil(Int, island_size * config.migration_rate))

    targets = migration_targets(n_islands, config.migration_topology, rng)

    # Collect emigrants from every island before modifying anything.
    # This prevents a sending island from receiving its own migrants back
    # in a single migration step (order-independence).
    emigrants = Vector{Vector{Individual}}(undef, n_islands)
    for src in 1:n_islands
        sorted = sortperm(islands[src]; by = ind -> ind.fitness, rev = true)
        top    = sorted[1:min(n_migrants, island_size)]
        emigrants[src] = [_deep_copy_individual(islands[src][k]) for k in top]
    end

    # Deliver emigrants to targets, replacing worst residents
    for src in 1:n_islands
        dst = targets[src]
        dst == src && continue   # self-send: no-op

        incoming = emigrants[src]
        island   = islands[dst]

        # Indices of the worst individuals on the receiving island
        sorted_asc = sortperm(island; by = ind -> ind.fitness, rev = false)
        replace_idx = sorted_asc[1:min(length(incoming), island_size)]

        for (k, idx) in enumerate(replace_idx)
            island[idx] = incoming[k]
        end
    end

    nothing
end


# =============================================================================
# 4. Private helpers
# =============================================================================

"""
    _deep_copy_individual(ind) → Individual

Return a fully independent copy of `ind`. All trees and arrays are cloned
so mutations on the copy do not affect the original.
"""
function _deep_copy_individual(ind::Individual)::Individual
    Individual(
        copy_tree.(ind.trees),
        copy(ind.weights),
        ind.combination,
        ind.fitness,
        ind.complexity,
        ind.age,
        copy(ind.behavior),
        ind.dominated,
        ind.pps_score,
        ind.rre_score,
    )
end

