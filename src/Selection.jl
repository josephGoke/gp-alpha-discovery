# =============================================================================
# Selection.jl — Population selection operators
#
# The selection method is determined entirely by `config.selection`:
#
#   :epsilon_lexicase  →  epsilon_lexicase_select
#   :tournament        →  tournament_select
#
# Both functions share the same call signature so Engine.jl can dispatch
# through a single `select` call without branching.
#
# Public API
# ──────────────────────────────────────────────────────────────────────────
#   tournament_select(pool, config, rng)                     → Individual
#   epsilon_lexicase_select(pool, errors, config, rng)       → Individual
#   select(pool, errors, config, rng)                        → Individual
#   select_n(pool, errors, n, config, rng)                   → Vector{Individual}
#
# Argument conventions
# ──────────────────────────────────────────────────────────────────────────
#   pool    : Vector{Individual} — the candidate pool (island sub-population)
#   errors  : Matrix{Float64}   — size (n_individuals × n_samples)
#               errors[i, j] = absolute prediction error of individual i on
#               sample j.  Pre-computed once per generation by the engine and
#               passed in — Selection.jl never calls eval_tree.
#   config  : GPConfig
#   rng     : AbstractRNG — caller-owned, seeded per island/generation
# =============================================================================

using Random

# =============================================================================
# 1. Tournament selection
# ============================================================================

"""
    tournament_select(pool, config, rng) → Individual

Sample 'config.tournamentk' individuals uniformly at random(with replacement) from 'pool' and return the one with the best fitness.

Requires only scalar fitness - 'errors' are not used.
"""

function tournament_select(pool::Vector{Individual}, config::GPConfig, rng::AbstractRNG)::Individual

    n = length(pool)
    k = config.tournament_k
    if k >= n
        indices = randperm(rng, n)
        best = pool[indices[1]]
        for idx in indices[2:end]
            pool[idx].fitness > best.fitness && (best = pool[idx])
        end
        return best

    else
        best = pool[rand(rng, 1:n)]
        for _ in 2:k
            candidate = pool[rand(rng, 1:n)]
            candidate.fitness > best.fitness && (best = candidate)
        end
        return best
    end
end


# =============================================================================
# 2. Epsilon-lexicase selection
# =============================================================================

"""
    epsilon_lexicase_select(pool, errors, config, rng) → Individual

Epsilon-lexicase selection for continuous regression (La Cava et al.).

Algorithm
─────────
1. Start with the full pool as candidates.
2. Shuffle the sample (column) order randomly.
3. For each sample j in shuffled order:
   a. Find the minimum error in the current candidate set on sample j.
   b. Retain only candidates whose error is within
      `config.lexicase_epsilon` of that minimum.
   c. If only one candidate remains, return it immediately.
4. If multiple candidates survive all cases, return one at random.

`errors[i, j]` must be the error of pool[i] on sample j.
"""

function epsilon_lexicase_select(pool::Vector{Individual}, errors::Matrix{Float64}, config::GPConfig, rng::AbstractRNG)::Individual


    n_ind = length(pool)
    n_periods = size(errors, 2)
    eps = config.lexicase_epsilon

    candidates = collect(1:n_ind)  # candidate indices

    period_order = randperm(rng, n_periods)  # random period order

    for k in period_order
        best_error = minimum(errors[i, k] for i in candidates)
        filter!(i -> errors[i, k] <= best_error + eps, candidates)

    end
    return pool[rand(rng, candidates)]
    
end


# =============================================================================
# 3. Unified selection interface
# =============================================================================

"""
    select(pool, errors, config, rng) → Individual

Dispatch to the configured selection method. 'errors' is ignored by tournament selection.
"""
function  select(pool::Vector{Individual}, errors::Matrix{Float64}, config::GPConfig, rng::AbstractRNG)::Individual
    if config.selection == :epsilon_lexicase
        return epsilon_lexicase_select(pool, errors, config, rng)
    elseif config.selection === :tournament
        return tournament_select(pool, config, rng)
    else
        error("Unknown selection method: $(config.selection)" * "Valid: :epsilon_lexicase, :tournament")
    end
end



"""
    select_n(pool, errors, n, config, rng) → Vector{Individual}

Run `select` n times to get a vector of selected individuals.
"""
function select_n(pool::Vector{Individual}, errors::Matrix{Float64}, n::Int, config::GPConfig, rng::AbstractRNG)::Vector{Individual}
    return [select(pool, errors, config, rng) for _ in 1:n]

end


# =============================================================================
# 4. Period-aggregated error matrix
# =============================================================================

"""
    aggregate_errors(errors, period_indices) → Matrix{Float64}

Compute the per-period mean absolute prediction error for every individual.

Returns a  '(n_individuals * n_periods)' matrix where 'errors[i, k]' = mean | signal_i[j] - y_raw[j]| over all rows j in period k. 

Period layout
─────────────
`n_samples` rows are divided into `config.lexicase_n_periods` contiguous
blocks of equal size.  If `n_samples` is not divisible, the last block
absorbs the remainder — it is never truncated.

All rows within a period retain their original temporal order.
Only period indices (not row indices) are shuffled by the selector.

Called once per generation by Engine.jl before selection.
Thread-safe: each individual writes to its own row of `errors`.
"""

function build_error_matrix(population::Vector{Individual}, X::Matrix{Float64}, y_raw::Vector{Float64}, config::GPConfig)::Matrix{Float64}

    n_ind = length(population)
    n_samples = length(y_raw)
    
    n_periods = min(config.lexicase_n_periods, n_samples)
    block_size = div(n_samples, n_periods)
    errors = Matrix{Float64}(undef, n_ind, n_periods)
   
    Threads.@threads for i in 1:n_ind
        signal = combine_trees(population[i], X)
        abs_err = abs.(signal .- y_raw)

        for k in 1:n_periods
            lo = (k - 1) * block_size + 1
            hi = k == n_periods ? n_samples : k * block_size
            errors[i, k] = mean(abs_err[lo:hi])
        end
    end
    return errors
end