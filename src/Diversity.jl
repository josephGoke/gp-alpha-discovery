# =============================================================================
# Diversity.jl — Novelty archive + 3-objective Pareto Hall of Fame
#
# Two complementary mechanisms that prevent the convergence failure seen in V1:
#
# 1. Novelty Archive (NoveltyArchive)
#    Rewards individuals for behaving differently from anything seen before.
#    "Behavior" = factor output on a fixed set of rows (behavioral fingerprint).
#    Novelty score = mean Euclidean distance to the K nearest neighbors in
#    behavior space across current population + archive.
#    Annealed: starts at full novelty_weight, decays to 0 over the final 25%
#    of the run (handled by current_novelty_weight in Fitness.jl).
#    Eviction policy: when archive is full, drop the least novel archive member
#    (the one whose mean distance to its K archive-neighbors is smallest).
#
# 2. Pareto Hall of Fame (ParetoHoF)
#    Tracks the Pareto frontier over 3 objectives:
#      PPS  (maximize) — Predictive Power Score
#      RRE  (maximize) — Relative Rank Entropy (signal spread / discriminability)
#      complexity (minimize) — total node count
#    An individual enters only if it is non-dominated on this frontier.
#    When hof_size is exceeded, members closest together in objective space
#    are pruned (crowding distance trimming).
#
# Public API
# ──────────────────────────────────────────────────────────────────────────
#   NoveltyArchive(max_size)
#   behavioral_distance(b1, b2)                              → Float64
#   knn_mean_distance(query, others, k)                      → Float64
#   compute_novelty_scores(population, archive, config)      → Vector{Float64}
#   update_archive!(archive, population, config)             → Nothing
#
#   ParetoHoF(max_size)
#   pareto_dominates(a, b)                                   → Bool
#   update_hof!(hof, population)                             → Nothing
#   hof_summary(hof)                                         → String
# =============================================================================

using Statistics


# =============================================================================
# 1. NoveltyArchive
# =============================================================================

"""
    NoveltyArchive

Stores the behavioral fingerprints of historically novel individuals.
Each entry is a `Vector{Float32}` matching `Individual.behavior`.

Fields
──────
- `behaviors` : archived behavior vectors (one per archived individual)
- `max_size`  : capacity; when full, least-novel entry is evicted on insert
"""
mutable struct NoveltyArchive
    behaviors::Vector{Vector{Float32}}
    max_size::Int
end

NoveltyArchive(max_size::Int) = NoveltyArchive(Vector{Vector{Float32}}(), max_size)


"""
    behavioral_distance(b1, b2) → Float64

Euclidean distance between two behavior vectors.
Uses Float64 accumulator for precision despite Float32 storage.
"""
function behavioral_distance(b1::Vector{Float32}, b2::Vector{Float32})::Float64
    length(b1) == length(b2) ||
        error("behavioral_distance: length mismatch ($(length(b1)) vs $(length(b2)))")
    d = 0.0
    @inbounds for i in eachindex(b1)
        delta = Float64(b1[i]) - Float64(b2[i])
        d += delta * delta
    end
    return sqrt(d)
end


"""
    knn_mean_distance(query, others, k) → Float64

Mean Euclidean distance from `query` to its `k` nearest neighbors in `others`.
Returns 0.0 when `others` is empty.
Brute-force O(|others|) sort — adequate for archive sizes ≤ novelty_archive_size + pop_size.
"""
function knn_mean_distance(query::Vector{Float32},
                            others::Vector{Vector{Float32}},
                            k::Int)::Float64
    isempty(others) && return 0.0
    k_actual = min(k, length(others))
    dists = [behavioral_distance(query, o) for o in others]
    sort!(dists)
    return mean(dists[1:k_actual])
end


"""
    compute_novelty_scores(population, archive, config) → Vector{Float64}

Compute a novelty score for every individual in `population`.

Score = mean Euclidean distance to K nearest neighbors in the combined space
of (all other population behaviors) ∪ (archive behaviors).

Individuals with empty behavior vectors receive score 0.0.
"""
function compute_novelty_scores(population::Vector{Individual},
                                 archive::NoveltyArchive,
                                 config::GPConfig)::Vector{Float64}
    n      = length(population)
    k      = config.novelty_k
    scores = Vector{Float64}(undef, n)

    arch_behaviors = archive.behaviors

    for i in 1:n
        query = population[i].behavior
        if isempty(query)
            scores[i] = 0.0
            continue
        end

        # Pool = all other population members + all archive entries
        pool = Vector{Vector{Float32}}()
        sizehint!(pool, n - 1 + length(arch_behaviors))
        for j in 1:n
            j == i && continue
            isempty(population[j].behavior) && continue
            push!(pool, population[j].behavior)
        end
        append!(pool, arch_behaviors)

        scores[i] = knn_mean_distance(query, pool, k)
    end

    return scores
end


"""
    _archive_self_novelty(behaviors, idx, k) → Float64

Novelty of archive entry `idx` relative to all other archive entries.
Used to identify the least-novel member for eviction.
"""
function _archive_self_novelty(behaviors::Vector{Vector{Float32}},
                                idx::Int, k::Int)::Float64
    pool = [behaviors[j] for j in eachindex(behaviors) if j != idx]
    return knn_mean_distance(behaviors[idx], pool, k)
end


"""
    update_archive!(archive, population, config) → Nothing

Consider each individual in `population` for admission to the novelty archive.

Admission rule: an individual's novelty score (vs current archive) must exceed
the mean novelty of existing archive members. This ensures we only archive
individuals that genuinely expand the explored behavior space.

When the archive is at capacity, the least-novel existing member is evicted
before the new entry is added.

Individuals with empty behavior vectors are skipped.
"""
function update_archive!(archive::NoveltyArchive,
                          population::Vector{Individual},
                          config::GPConfig)
    k = config.novelty_k

    for ind in population
        isempty(ind.behavior) && continue

        # Novelty of this candidate vs current archive
        nov = knn_mean_distance(ind.behavior, archive.behaviors, k)

        # Admission threshold: must exceed mean internal archive novelty
        # (empty archive → always admit)
        if isempty(archive.behaviors)
            threshold = -Inf
        else
            arch_novs = [_archive_self_novelty(archive.behaviors, j, k)
                         for j in eachindex(archive.behaviors)]
            threshold = mean(arch_novs)
        end

        nov > threshold || continue

        # Evict least-novel archive member if at capacity
        if length(archive.behaviors) >= archive.max_size
            arch_novs = [_archive_self_novelty(archive.behaviors, j, k)
                         for j in eachindex(archive.behaviors)]
            deleteat!(archive.behaviors, argmin(arch_novs))
        end

        push!(archive.behaviors, copy(ind.behavior))
    end

    nothing
end


# =============================================================================
# 2. ParetoHoF — 3-objective Hall of Fame (PPS↑, RRE↑, complexity↓)
# =============================================================================

"""
    ParetoHoF

Pareto Hall of Fame tracking the non-dominated frontier over three objectives:
  - PPS        (maximize) — Predictive Power Score
  - RRE        (maximize) — Relative Rank Entropy (signal spread)
  - complexity (minimize) — total node count (parsimony proxy)

Fields
──────
- `members`  : non-dominated individuals on the current frontier
- `max_size` : capacity; excess members trimmed by crowding distance
"""
mutable struct ParetoHoF
    members::Vector{Individual}
    max_size::Int
end

ParetoHoF(max_size::Int) = ParetoHoF(Individual[], max_size)


"""
    pareto_dominates(a, b) → Bool

Return true if `a` weakly dominates `b` on all three objectives with at
least one strict improvement.

  PPS        : a.pps_score  ≥ b.pps_score
  RRE        : a.rre_score  ≥ b.rre_score
  complexity : a.complexity ≤ b.complexity  (smaller is better)
"""
function pareto_dominates(a::Individual, b::Individual)::Bool
    (a.pps_score  >= b.pps_score  &&
     a.rre_score  >= b.rre_score  &&
     a.complexity <= b.complexity) &&
    (a.pps_score  >  b.pps_score  ||
     a.rre_score  >  b.rre_score  ||
     a.complexity <  b.complexity)
end


"""
    update_hof!(hof, population) → Nothing

Attempt to add each individual in `population` to the Hall of Fame.

Per-candidate:
1. Skip if dominated by any current HoF member.
2. Add the candidate (it is non-dominated).
3. Remove any HoF members now dominated by the new entry.

After processing all candidates, trim to `hof.max_size` using crowding
distance: drop the member with the smallest nearest-neighbor distance in
3-objective space (most redundant point on the frontier).

Sets `ind.dominated = false` for admitted members.
"""
function update_hof!(hof::ParetoHoF, population::Vector{Individual})
    for candidate in population
        # Skip unevaluated individuals
        candidate.fitness == -Inf && continue

        # Rejected if dominated by any current member
        any(pareto_dominates(m, candidate) for m in hof.members) && continue
        
        # Skip if an objectively identical entry already exists
        # (prevents double-insertion when the same individual is re-submitted)
        any(hof.members) do m
            m.pps_score  == candidate.pps_score  &&
            m.rre_score  == candidate.rre_score  &&
            m.complexity == candidate.complexity
        end && continue
        
        # Admitted
        push!(hof.members, candidate) 
        candidate.dominated = false

        # Evict members now dominated by the new entry
        filter!(m -> !pareto_dominates(candidate, m), hof.members)
    end

    _trim_hof!(hof)
    nothing
end


"""
    _trim_hof!(hof) → Nothing

Reduce `hof.members` to `hof.max_size` by repeatedly dropping the member
with the smallest crowding distance (nearest neighbor in 3-objective space).
Keeps the frontier maximally spread.
"""
function _trim_hof!(hof::ParetoHoF)
    length(hof.members) <= hof.max_size && return

    while length(hof.members) > hof.max_size
        n = length(hof.members)
        min_dist = Inf
        drop_idx = 1

        for i in 1:n
            a = hof.members[i]
            for j in (i+1):n
                b = hof.members[j]
                max_cmplx = max(a.complexity, b.complexity, 1)
                d = (a.pps_score  - b.pps_score)^2  +
                    (a.rre_score  - b.rre_score)^2  +
                    ((a.complexity - b.complexity) / max_cmplx)^2
                if d < min_dist
                    min_dist = d
                    # Of the closest pair, drop the one with lower PPS
                    drop_idx = a.pps_score <= b.pps_score ? i : j
                end
            end
        end

        hof.members[drop_idx].dominated = true
        deleteat!(hof.members, drop_idx)
    end

    nothing
end


"""
    hof_summary(hof) → String

One-line summary of the HoF for per-generation logging.
"""
function hof_summary(hof::ParetoHoF)::String
    isempty(hof.members) && return "HoF: empty"
    best_pps  = maximum(m.pps_score  for m in hof.members)
    best_rre  = maximum(m.rre_score  for m in hof.members)
    min_cmplx = minimum(m.complexity for m in hof.members)
    max_cmplx = maximum(m.complexity for m in hof.members)
    return "HoF: $(length(hof.members)) members | " *
           "best_pps=$(round(best_pps,  digits=4)) | " *
           "best_rre=$(round(best_rre,  digits=4)) | " *
           "complexity=$(min_cmplx)–$(max_cmplx)"
end