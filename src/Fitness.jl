# =============================================================================
# Fitness.jl — IC, ICIR, novelty annealing, fitness scoring, population eval
#
# Public API
# ─────────────────────────────────────────────────────────────────────────────
#   prerank_y(y_raw)                            → Vector{Float64}
#   rank_ic(factor, y_ranked)                   → Float64
#   icir(factor, y_raw [, n_periods])           → Float64
#   current_novelty_weight(config, gen)         → Float64
#   compute_fitness(signal, y_ranked,
#                   complexity, config; ...)    → Float64
#   evaluate_fitness!(ind, X_sub, y_raw,
#                     y_ranked, config; ...)   → Float64
#   evaluate_population!(population, X, y_raw,
#                        behavior_rows,
#                        config, rng, gen)      → Nothing
# =============================================================================


using Statistics
using StatsBase: tiedrank


# ===============================================================================
# 1. Preranking 
# ===============================================================================
"""
    prerank_y(y_raw) -> Vector{Float64}

Compute the tied rank of target values 'y_raw' once and cache the result.
Pass the returned vector to every 'spearman_ic' call to avoid redundant ranking computations.
"""

prerank_y(y_raw::Vector{Float64})::Vector{Float64} = tiedrank(y_raw)



# ===============================================================================
# 2. Per-individual metrics (Fitness Metrics)
# ===============================================================================
""" 
    ic(factor, y_raw) -> Float64

Information Coefficient: Pearson Correlation between the 'factor' and 'y_raw'.
"""
function ic(factor::Vector{Float64}, y_raw::Vector{Float64})::Float64
    length(factor) == length(y_raw) || error("Pearson IC: length mismatch - factor=$(length(factor)), y=$(length(y_raw))")
    isempty(factor) && return 0.0

    # Handle constant vectors causing NaN in cor()
    std_f = std(factor)
    std_y = std(y_raw)
    (std_f < 1e-8 || std_y < 1e-8) && return 0.0

    return cor(factor, y_raw)
end

"""
    rank_ic(factor, y_ranked) -> Float64

Rank-order (Spearman) correlation between the 'factor' and pre-ranked 'y_ranked'.
Returns 0.0 when factor is constant.
"""
function rank_ic(factor::Vector{Float64}, y_ranked::Vector{Float64})::Float64
    length(factor) == length(y_ranked) || error("Spearman IC: length mismatch - factor=$(length(factor)), y_ranked=$(length(y_ranked))")
    isempty(factor) && return 0.0

    f_ranked = tiedrank(factor)
    std_f = std(f_ranked)
    std_y = std(y_ranked)  # Actually constant for a fixed y_ranked, but safe to check
    (std_f < 1e-8 || std_y < 1e-8) && return 0.0
    return cor(f_ranked, y_ranked)
end

"""
    icir(factor, y_raw [, n_periods]) -> Float64

IC Information Ratio: mean(IC per period) / std(IC per period)
"""
function icir(factor::Vector{Float64}, y_raw::Vector{Float64}, n_periods::Int = 20)::Float64
    n = length(factor)
    n_periods < 2 && return 0.0
    period_size = floor(Int, n / n_periods)
    period_size < 2 && return 0.0

    ics = Vector{Float64}(undef, n_periods)
    @inbounds for p in 1:n_periods
        lo = (p - 1) * period_size + 1
        hi = p * period_size
        yr_win = tiedrank(y_raw[lo:hi])
        ics[p] = rank_ic(factor[lo:hi], yr_win)
    end

    mean_ics = mean(ics)
    std_ics = std(ics)
    return std_ics < 1e-8 ? 0.0 : mean_ics / std_ics
end

"""
    rolling_rank_ic(factor, y_raw, n_periods) -> Vector{Float64}

Returns the array of IC values across time split into n_periods.
"""
function rolling_rank_ic(factor::Vector{Float64}, y_raw::Vector{Float64}, n_periods::Int = 20)::Vector{Float64}
    n = length(factor)
    if n_periods < 2 || floor(Int, n / n_periods) < 2
        return Float64[]
    end
    period_size = floor(Int, n / n_periods)

    rolling_ics = Vector{Float64}(undef, n_periods)
    @inbounds for p in 1:n_periods
        lo = (p - 1) * period_size + 1
        hi = p * period_size
        y_ranked = tiedrank(y_raw[lo:hi])
        rolling_ics[p] = rank_ic(factor[lo:hi], y_ranked)
    end
    
    return rolling_ics
end


# =============================================================================
# Predictive Power Score (PPS)
# =============================================================================

"""
    pps(factor, y_raw) -> Float64

Predictive Power Score (PPS) = β · IC + (1 - β) · RankIC
A composite metric combining IC and Rank IC, weighted by β.
Measures the predictive strength of an alpha across time
A higher PPS indicates stronger alignment between the alpha and subsequent asset returns.
"""
function pps(factor::Vector{Float64}, y_raw::Vector{Float64}, y_ranked::Vector{Float64}, β::Float64 = 0.5)::Float64
    length(factor) == length(y_ranked) || error("PPS: length mismatch - factor=$(length(factor)), y_ranked=$(length(y_ranked))")
    isempty(factor) && return 0.0

    p_ic = ic(factor, y_raw)


    # y_ranked = tiedrank(y_raw)
    spearman_ic = rank_ic(factor, y_ranked)

    return β * p_ic + (1.0 - β) * spearman_ic 
end



# =============================================================================
# Relative Rank Entropy (RRE)
# =============================================================================

"""
    rre(signal; n_bins=10) → Float64

Relative Rank Entropy — measures how non-uniformly a factor distributes its
cross-sectional signal, on a scale of 0 to 1.

Intuition
─────────
A factor that assigns identical or near-identical scores to all assets has
a flat (uniform) rank distribution — maximum entropy, zero discriminability.
A factor with extreme long and short positions concentrates mass at the tails,
producing a non-uniform distribution and lower entropy.

RRE = 1 - H(empirical) / H(uniform)

where H(uniform) = log(n_bins) is the maximum possible entropy for n_bins bins.

  RRE = 0  →  perfectly uniform rank distribution (random signal)
  RRE = 1  →  all mass in one bin (perfectly concentrated signal)

A useful alpha sits in the middle — neither random (RRE≈0) nor pathologically
concentrated (RRE≈1, which may indicate look-ahead or data leakage).

Arguments
─────────
- `signal`  : evaluated factor vector (pre-eval by combine_trees)
- `n_bins`  : number of histogram bins; 10 is standard for moderate n
"""
function rre(signal::Vector{Float64}; n_bins::Int = 10)::Float64
    n = length(signal)
    n < n_bins && return 0.0       # too few observations to bin meaningfully

    # Normalise ranks to (0, 1]
    ranks = tiedrank(signal) ./ n

    # Bin counts
    bin_counts = zeros(Int, n_bins)
    @inbounds for r in ranks
        b = min(ceil(Int, r * n_bins), n_bins)
        bin_counts[b] += 1
    end

    # Empirical entropy (nats; skip empty bins — log(0) undefined)
    H = 0.0
    @inbounds for c in bin_counts
        c == 0 && continue
        p = c / n
        H -= p * log(p)
    end

    H_max = log(Float64(n_bins))    # entropy of a uniform distribution over n_bins
    H_max < 1e-10 && return 0.0     # degenerate case: n_bins = 1

    return 1.0 - H / H_max
end


# =============================================================================
# Perturbation Fidelity Score (PFS)
# =============================================================================

"""
    pfs(ind, X; rng, n_perturbations=5, noise_scale=0.01) → Float64

Perturbation Fidelity Score — measures the robustness of an individual's
factor signal under small Gaussian perturbations of the input data.

Intuition
─────────
A factor that changes its rank ordering dramatically when inputs are perturbed
by 1% noise is likely overfit or numerically brittle — it found a pattern
specific to the training data rather than a robust structural relationship.
PFS quantifies this stability.

Algorithm
─────────
1. Evaluate the original signal: s₀ = combine_trees(ind, X).
2. For k = 1…n_perturbations:
     Xₖ = X + ε,  ε ~ N(0, noise_scale² · σ²_col)  (column-relative noise)
     sₖ = combine_trees(ind, Xₖ)
     ρₖ = Spearman rank correlation between s₀ and sₖ
3. PFS = mean(ρ₁, …, ρₙ)

  PFS ~ 1.0  →  highly stable signal (same rank order under noise)
  PFS ~ 0.0  →  signal collapses to random under tiny perturbations
  PFS < 0.0  →  signal flips — structurally unstable 

Column-relative noise (scaled to each feature's standard deviation) ensures
that the perturbation is equally "small" regardless of feature scale.


Arguments
─────────
- `ind`             : Individual to evaluate
- `X`               : data matrix (n_rows * n_features)
- `rng`             : seeded RNG — for reproducibility
- `n_perturbations` : number of noise realizations;
- `noise_scale`     : noise magnitude as fraction of each column's std (default 1%)
"""
function pfs(ind::Individual, X::Matrix{Float64};
             rng::AbstractRNG = MersenneTwister(0),
             n_perturbations::Int = 5,
             noise_scale::Float64 = 0.01)::Float64

    n_rows, n_cols = size(X)

    # Original signal
    s0 = combine_trees(ind, X)
    std_s0 = std(s0)
    std_s0 < 1e-8 && return 0.0     # constant signal — PFS undefined, treat as 0

    # Pre-compute per-column standard deviations for relative scaling
    col_stds = [std(X[:, j]) for j in 1:n_cols]
    col_stds .= max.(col_stds, 1e-8)   # floor to avoid zero-std columns

    correlations = 0.0
    for _ in 1:n_perturbations
        X_perturbed = copy(X)
        for j in 1:n_cols
            X_perturbed[:, j] .+= randn(rng, n_rows) .* (noise_scale * col_stds[j])
        end

        s_perturbed = combine_trees(ind, X_perturbed)

        # Rank correlation: robust to signal rescaling
        p = cor(tiedrank(s0), tiedrank(s_perturbed))
        correlations += isfinite(p) ? p : 0.0
    end

    return correlations / n_perturbations
end












# =================================================================================
# 3. Novelty weight annealing
# =================================================================================

"""
    current_novelty_weight(config, gen) -> Float64

Linearly anneal `config.fitness_weights.novelty` → 0.0 over the final 25% of the run.

  gen ≤ 75% of n_generations  →  config.fitness_weights.novelty   (full exploration)
  gen = n_generations         →  0.0                     (pure exploitation)
  in between                  →  linear interpolation

Keeps exploration pressure high during the main search phase and switches to pure exploitation at the end, preventing novelty from disrupting
convergence on the best discovered signals.
"""

function  current_novelty_weight(config::GPConfig, gen::Int)::Float64
    w = config.fitness_weights.novelty
    w == 0.0 && return 0.0

    anneal_start = floor(Int, 0.75 * config.n_generations)
    gen <= anneal_start && return w
    frac = (gen - anneal_start) / max(config.n_generations - anneal_start, 1)
    return w * (1.0 - clamp(frac, 0.0, 1.0))
    
end


# ===============================================================================
# 4. Fitness scoring
# ===============================================================================
"""
    compute_fitness(signal, y_ranked, complexity, config; novelty, rre_val, pfs_val) -> Float64

Compute scalar fitness for a pre-evaluate factor signal

  fitness = (1 - w) * IC  +  w * novelty  -  parsimony * complexity

where w = `novelty_weight` (passed in after annealing by the caller).

`novelty` and `novelty_weight` default to 0.0 — Diversity.jl supplies them
"""

function compute_fitness(factor::Vector{Float64}, y_raw::Vector{Float64},
                        y_ranked::Vector{Float64},
                         complexity::Int, config::GPConfig;
                         novelty::Float64 = 0.0,
                         rre_val::Float64 = 0.0,
                         pfs_val::Float64 = 0.0)::Float64
    fw = config.fitness_weights
    pps_val = pps(factor, y_raw, y_ranked)
    penalty = config.parsimony * complexity

    return fw.pps * pps_val + fw.rre * rre_val + fw.pfs * pfs_val + fw.novelty * novelty - penalty
end




"""
    evaluate_fitness!(ind, X_sub, y_ranked_sub, config;
                      novelty, novelty_weight) → Float64

Evaluate `ind` on the subsampled data `(X_sub, y_ranked_sub)`, mutate
`ind.fitness`, and return the new fitness value.

`y_ranked_sub` must be `tiedrank(y_raw[sub_idx])` — ranked on the **same**
row subset as `X_sub`.
"""
function evaluate_fitness!(ind::Individual, X_sub::Matrix{Float64}, y_raw::Vector{Float64}, y_ranked::Vector{Float64}, config::GPConfig; novelty::Float64 = 0.0, pfs_val::Float64 = 0.0)::Float64

    signal  = combine_trees(ind, X_sub)
    rre_val = rre(signal)
    pps_val = pps(signal, y_raw, y_ranked)

    ind.pps_score = pps_val
    ind.rre_score = rre_val
    
    ind.fitness = compute_fitness(signal, y_raw, y_ranked, ind.complexity, config;
                                  novelty = novelty, rre_val = rre_val, pfs_val = pfs_val)
    ind.fitness
end



# ======================================================================================
# 5. Population evaluation (called once per generation by Engine.jl)
# ======================================================================================
"""
    evaluate_population!(population, X, y_raw, behavior_rows, config, rng, gen)

Full per-generation fitness pass over 'population'

Steps
1. **Subsample rows** — draw `floor(n_rows * eval_subsample)` rows as a
   contiguous block (random start index). Contiguous sampling preserves
   temporal order, which matters for TS operators. A fresh block each
   generation adds stochasticity that helps escape plateaus.

2. **Re-rank y on the subsample** — one `tiedrank` call shared across all
   individuals.  This is the V1 bottleneck fix: O(n_sub log n_sub) once
   instead of O(n log n) * pop_size.

3. **Parallel fitness eval** — `Threads.@threads` over individuals.
   Each thread reads from the immutable `(X_sub, y_ranked_sub)` and writes
   only to its own `ind` — no shared mutable state, no locks needed.

4. **Behavior fingerprints** — evaluated on fixed `behavior_rows` (not the
   subsampled rows).  Using fixed rows keeps behavioral distances comparable
   across all generations and the novelty archive.

5. **Complexity update** — recomputed for each individual after structural
   changes; cheap vs the eval cost so always done here.


Arguments:

- `population`    : `Vector{Individual}`
- `X`             : full training matrix (n_rows * n_features)
- `y_raw`         : raw target vector — needed for per-subsample re-ranking
- `behavior_rows` : fixed row indices for behavior fingerprinting (startup)
- `config`        : `GPConfig`
- `rng`           : seeded coordinator RNG (subsampling only — not threaded)
- `gen`           : current generation number (novelty weight annealing)
"""

function evaluate_population!(population::Vector{Individual},
                               X::Matrix{Float64},
                               y_raw::Vector{Float64},
                               behavior_rows::Vector{Int},
                               config::GPConfig,
                               rng::AbstractRNG,
                               gen::Int)
    n_rows = size(X, 1)
    n_sub  = max(1, floor(Int, n_rows * config.eval_subsample))

    # Step 1 — subsample (once per generation)
    start_idx = rand(rng, 1:(n_rows - n_sub + 1))
    sub_idx = start_idx:(start_idx + n_sub - 1)
    X_sub = X[sub_idx, :]

    # Step 2 - ranking
    y_raw_sub = y_raw[sub_idx]
    y_ranked_sub = tiedrank(y_raw_sub)

    nw = current_novelty_weight(config, gen)

    # Steps 3
    Threads.@threads for ind in population
        update_complexity!(ind)
        evaluate_fitness!(ind, X_sub, y_raw_sub, y_ranked_sub, config; novelty = nw)
    end

    # Step 5 — PFS on elites only, every N generations
    if config.fitness_weights.pfs > 0.0 &&
       gen % config.const_opt_every_n_gens == 0

        order   = sortperm(population; by = ind -> ind.fitness, rev = true)
        n_elite = min(config.elite_count, length(population))
        elites  = order[1:n_elite]

        Threads.@threads for i in elites
            ind     = population[i]
            # Deterministic per-(gen, position) seed — no shared mutable RNG
            p_rng   = MersenneTwister(gen * 100_000 + i)
            pfs_val = pfs(ind, X_sub;
                          rng             = p_rng,
                          n_perturbations = config.pfs_n_perturbations,
                          noise_scale     = config.pfs_noise_scale)
            # Re-score with PFS included; RRE recomputed (signal not cached)
            signal  = combine_trees(ind, X_sub)
            rre_val = rre(signal)
            ind.fitness = compute_fitness(signal, y_raw_sub, y_ranked_sub,
                                          ind.complexity, config;
                                          novelty  = nw,
                                          rre_val  = rre_val,
                                          pfs_val  = pfs_val)
        end
    end


    # Step 5 — behavior fingerprints on fixed rows
    Threads.@threads for ind in population
        update_behavior!(ind, X, behavior_rows)
    end

    nothing
end
