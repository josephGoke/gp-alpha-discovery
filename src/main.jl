# =============================================================================
# main.jl — GP Alpha Discovery: load → train → run → evaluate OOS
#
# Usage
# ──────────────────────────────────────────────────────────────────────────
#   julia --project --threads=4 src/main.jl
#
# Data expectations
# ──────────────────────────────────────────────────────────────────────────
#   DATA_DIR contains one CSV per asset.
#   Each CSV has columns:  [date_col, feature_1, ..., feature_n, target_col]
#     - First column  : date (any parseable format; used for sorting + split)
#     - Middle columns: numeric features (all used as GP inputs)
#     - Last column   : numeric target (forward return to predict)
#   Assets are concatenated vertically in alphabetical file order,
#   time-ordered within each asset.
#
# Train / test split
# ──────────────────────────────────────────────────────────────────────────
#   TRAIN_FRAC = 0.8  →  first 80% of unique dates = train, rest = test.
#   All rows with a date ≤ cutoff_date go to train; the rest to test.
#   This prevents look-ahead bias: test rows are always strictly after
#   all training rows of the same date.
#
# Output
# ──────────────────────────────────────────────────────────────────────────
#   - Console: per-generation log + OOS metrics table
#   - results/hof_signals.csv   : signal values for every HoF member on test set
#   - results/hof_summary.csv   : PPS, IC, RankIC, RRE, complexity per member
# =============================================================================

using CSV
using DataFrames
using Statistics
using Dates
using Printf
using Random

include("Genetic.jl")
using .Genetic


# =============================================================================
# Configuration — edit these before running
# =============================================================================

const DATA_DIR   = "data"          # directory containing per-asset CSVs
const RESULTS_DIR = "results"      # output directory (created if absent)

# ── GP hyperparameters (tuned for a 4-core laptop) ───────────────────────────
const GP_CFG = GPConfig(
    # Population
    population_size        = 100,      # divisible by n_islands=4
    n_trees_per_individual = 13,
    combination            = :rank_average,

    # Tree shape
    min_depth              = 2,
    max_depth              = 6,

    # Evolution
    n_generations          = 20,
    elite_count            = 10,
    crossover_prob         = 0.85,
    mutation_prob          = MutationProbs(0.40, 0.35, 0.15, 0.10),

    # Selection
    selection              = :epsilon_lexicase,
    tournament_k           = 5,
    lexicase_epsilon       = 0.05,
    lexicase_n_periods     = 20,
    
    # Constant optimisation
    const_opt_method       = :nelder_mead,
    const_opt_every_n_gens = 5,
    
    # Fitness
    parsimony              = 0.001,
    fitness_weights        = FitnessWeights(0.5, 0.3, 0.1, 0.1),

    # Diversity
    novelty_k              = 10,
    novelty_archive_size   = 100,
    behavior_sample_size   = 200,
    hof_size               = 20,
    pfs_noise_scale        = 0.01,
    pfs_n_perturbations    = 3,

    # Islands — 4 islands matches 4 cores
    n_islands              = 4,
    migration_interval     = 5,
    migration_rate         = 0.05,
    migration_topology     = :ring,

    # Operators — full suite
    unary_op_names  = [:safe_sqrt, :safe_log, :safe_inv, :safe_neg,
    :safe_sin, :safe_cos, :safe_tanh, :abs],
    binary_op_names = [:safe_add, :safe_sub, :safe_mul, :safe_div,
    :safe_pow, :signed_power],
    ts_op_names     = [:ts_delta, :ts_mean, :ts_stddev, :ts_rank,
                       :ts_max, :ts_min, :ts_sum, :decay],
    ts_binary_op_names = [:ts_corr, :ts_cov],
    cs_op_names     = [:cs_rank, :cs_zscore, :cs_scale, :cs_winsorize],
    ts_window_sizes = [3, 5, 10, 20],

    const_prob   = 0.0,
    fin_op_prob  = 0.5,

    # Execution
    seed           = 42,
    eval_subsample = 0.3,    # 30% of rows per generation — preserves TS context
    verbose        = true,
    )
    
    

const TRAIN_FRAC = 1 - GP_CFG.eval_subsample          # fraction of unique dates used for training
const RETURN_HORIZON = 3                             # how many rows forward the target return looks

    # =============================================================================
# 1. Data loading
# =============================================================================

"""
    load_asset_csv(path) → (X::Matrix{Float64}, y::Vector{Float64}, dates::Vector)

Load a single asset CSV. Assumes:
  - Column 1 : date
  - Columns 2:(end-1) : numeric features
  - Column end : numeric target
Rows with any missing value are dropped. Returns rows sorted by date.
"""
function load_asset_csv(path::String; period::Int=RETURN_HORIZON)
    df = CSV.read(path, DataFrame; missingstring=["", "NA", "NaN", "null"])
    dropmissing!(df)

    # :open_time in names(df) || error("$path is missing open_time")
    # :close in names(df) || error("$path is missing close")

    sort!(df, :open_time)

    ncols = ncol(df)
    ncols >= 3 || error("$path must have at least 3 columns (date, ≥1 feature, target)")

    close    = Float64.(df[!, :close])
    
    dates    = collect(df[1:end-period, :open_time])
    features = Matrix{Float64}(df[1:end-period, Not(:open_time)])
    
    target   = close[(1 + period):end] ./ close[1:end-period] .- 1.0
    


    return features, target, dates
end


"""
    load_all_assets(data_dir) → (X, y, dates, n_features, file_names)

Load all CSVs from `data_dir`, validate they have the same number of feature
columns, and concatenate vertically (assets stacked in alphabetical order,
time-ordered within each asset).
"""
function load_all_assets(data_dir::String, period::Int=RETURN_HORIZON)
    isdir(data_dir) || error("Data directory not found: '$data_dir'")

    csv_files = sort(filter(f -> endswith(f, ".csv"), readdir(data_dir; join=true)))
    isempty(csv_files) && error("No CSV files found in '$data_dir'")

    @info "Found $(length(csv_files)) asset files in '$data_dir'"

    all_X     = Matrix{Float64}[]
    all_y     = Vector{Float64}[]
    all_dates = []

    n_features = nothing
    for path in csv_files
        X, y, dates = load_asset_csv(path; period = period)
        if n_features === nothing
            n_features = size(X, 2)
        else
            size(X, 2) == n_features ||
                error("Feature count mismatch in $(basename(path)): " *
                      "expected $n_features, got $(size(X, 2))")
        end
        push!(all_X, X)
        push!(all_y, y)
        push!(all_dates, collect(dates))
        @info "  $(basename(path)): $(size(X, 1)) rows, $n_features features"
    end

    X_full     = vcat(all_X...)
    y_full     = vcat(all_y...)
    dates_full = vcat(all_dates...)

    @info "Total: $(size(X_full, 1)) rows × $n_features features after stacking"
    return X_full, y_full, dates_full, n_features, basename.(csv_files)
end


# =============================================================================
# 2. Train / test split
# =============================================================================

"""
    train_test_split(X, y, dates, train_frac)
        → (X_train, y_train, X_test, y_test, cutoff_date)

Split by date: the first `train_frac` fraction of *unique* dates go to train,
the rest to test. Preserves temporal order within each split.
"""
function train_test_split(X::Matrix{Float64}, y::Vector{Float64},
                           dates, train_frac::Float64)
    unique_dates = sort(unique(dates))
    n_train_dates = max(1, floor(Int, length(unique_dates) * train_frac))
    cutoff = unique_dates[n_train_dates]

    train_mask = dates .<= cutoff
    test_mask  = .!train_mask

    X_train = X[train_mask, :]
    y_train = y[train_mask]
    X_test  = X[test_mask,  :]
    y_test  = y[test_mask]

    @info "Train/test split at date $cutoff"
    @info "  Train: $(sum(train_mask)) rows ($(round(100*mean(train_mask), digits=1))%)"
    @info "  Test:  $(sum(test_mask)) rows ($(round(100*mean(test_mask),  digits=1))%)"

    return X_train, y_train, X_test, y_test, cutoff
end


# =============================================================================
# 3. OOS evaluation
# =============================================================================


function evaluate_member_trees(
    m::Individual,
    X_test::Matrix{Float64},
    y_test::Vector{Float64},
    y_ranked_test::Vector{Float64},
    ind_rank::Int,
)::DataFrame
    rows = NamedTuple[]

    for (tree_idx, tree) in enumerate(m.trees)
        tree_signal = eval_tree(tree, X_test)

        push!(rows, (
            individual_rank = ind_rank,
            tree_idx        = tree_idx,
            oos_pps         = round(pps(tree_signal, y_test, y_ranked_test), digits = 4),
            oos_ic          = round(ic(tree_signal, y_test), digits = 4),
            oos_rankic      = round(rank_ic(tree_signal, y_ranked_test), digits = 4),
            oos_rre         = round(rre(tree_signal), digits = 4),
            tree_size       = tree.size,
            tree_depth      = tree.depth,
        ))
    end

    df = DataFrame(rows)
    sort!(df, :oos_pps, rev = true)
    df.tree_rank = 1:nrow(df)
    return df
end







"""
    evaluate_oos(hof, X_test, y_test) → DataFrame

Compute out-of-sample metrics for every HoF member:
  - PPS, IC (Pearson), RankIC (Spearman), RRE, complexity

Returns a DataFrame sorted by OOS PPS descending.
"""
function evaluate_oos(hof::ParetoHoF,
                       X_test::Matrix{Float64},
                       y_test::Vector{Float64})::Tuple{DataFrame, Vector{DataFrame}}
    isempty(hof.members) && return DataFrame(), DataFrame[]

    y_ranked_test = prerank_y(y_test)

    ind_rows = []
    tree_tables = DataFrame[]

    ranked_members = sort(hof.members; by = m -> m.pps_score, rev=true)

    for (ind_rank, m) in enumerate(ranked_members)
        signal   = combine_trees(m, X_test)
        oos_pps  = pps(signal, y_test, y_ranked_test)
        oos_ic   = ic(signal, y_test)
        oos_ric  = rank_ic(signal, y_ranked_test)
        oos_rre  = rre(signal)

        push!(ind_rows, (
            rank       = ind_rank,
            oos_pps    = round(oos_pps,  digits=4),
            oos_ic     = round(oos_ic,   digits=4),
            oos_rankic = round(oos_ric,  digits=4),
            oos_rre    = round(oos_rre,  digits=4),
            train_pps  = round(m.pps_score, digits=4),
            complexity = m.complexity,
        ))
        
        push!(tree_tables, evaluate_member_trees(m, X_test, y_test, y_ranked_test, ind_rank))
    end

    indiv_df = DataFrame(ind_rows)
    sort!(indiv_df, :oos_pps, rev=true)
    indiv_df.rank .= 1:nrow(indiv_df)


    return indiv_df, tree_tables
end










"""
    save_hof_signals(hof, X_test, results_dir) → Nothing

Write a CSV with one column per HoF member containing its OOS signal vector.
"""
function save_hof_signals(hof::ParetoHoF,
                           X_test::Matrix{Float64},
                           results_dir::String)
    isempty(hof.members) && return nothing

    signals = Dict{String, Vector{Float64}}()
    for (k, m) in enumerate(hof.members)
        signals["member_$(lpad(k, 2, '0'))_pps$(round(m.pps_score, digits=3))"] =
            combine_trees(m, X_test)
    end

    df = DataFrame(signals)
    path = joinpath(results_dir, "hof_signals.csv")
    CSV.write(path, df)
    @info "HoF signals saved to $path"
    nothing
end


# =============================================================================
# 4. Pretty-print helpers
# =============================================================================

function _print_oos_table(df::DataFrame)
    println("\n" * "="^72)
    println("  OUT-OF-SAMPLE RESULTS  ($(nrow(df)) HoF members)")
    println("="^72)
    @printf "  %-4s  %-8s  %-8s  %-8s  %-8s  %-8s  %s\n" "Rank" "OOS_PPS" "OOS_IC" "OOS_RIC" "OOS_RRE" "TRN_PPS" "Complexity"
    println("-"^72)
    for r in eachrow(df)
        @printf "  %-4d  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %d\n" r.rank r.oos_pps r.oos_ic r.oos_rankic r.oos_rre r.train_pps r.complexity
    end
    println("="^72)

    best = df[1, :]
    println("\n  Best OOS member:")
    @printf "    PPS=%.4f  IC=%.4f  RankIC=%.4f  RRE=%.4f  Complexity=%d\n" best.oos_pps best.oos_ic best.oos_rankic best.oos_rre best.complexity

    # Overfitting check
    if nrow(df) > 0
        avg_train = mean(df.train_pps)
        avg_test  = mean(df.oos_pps)
        degradation = avg_train > 0 ? (avg_train - avg_test) / avg_train * 100 : NaN
        @printf "\n  Mean train PPS: %.4f  →  mean OOS PPS: %.4f" avg_train avg_test
        isfinite(degradation) && @printf "  (%.1f%% degradation)" degradation
        println()
    end
    println()
end


function _print_tree_oos_table(df::DataFrame; max_rows::Int = 20)
    println("\n" * "="^72)
    println("  TREE-LEVEL OOS RESULTS  ($(nrow(df)) trees)")
    println("="^72)
    @printf "  %-4s  %-4s  %-8s  %-8s  %-8s  %-8s  %-8s  %-5s\n" "Ind" "Tree" "OOS_PPS" "OOS_IC" "OOS_RIC" "OOS_RRE" "Size" "Depth"
    println("-"^72)

    n = min(nrow(df), max_rows)
    for r in eachrow(first(df, n))
        @printf( "  %-4d  %-4d  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8d  %-5d\n",
            r.individual_rank, r.tree_idx, r.oos_pps, r.oos_ic, r.oos_rankic, r.oos_rre, r.tree_size, r.tree_depth
        )
    end

    if nrow(df) > max_rows
        println("  ... showing first $max_rows rows of $(nrow(df))")
    end
    println("="^72)
end

# =============================================================================
# 5. Entry point
# =============================================================================

function main()
    println("\n" * "="^72)
    println("  GP ALPHA DISCOVERY — V2")
    println("="^72)
    @info "Threads available: $(Threads.nthreads())"

    # ── Load data ─────────────────────────────────────────────────────────────
    X_full, y_full, dates_full, n_features, _ = load_all_assets(DATA_DIR, RETURN_HORIZON)

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test, _ =
        train_test_split(X_full, y_full, dates_full, TRAIN_FRAC)

    size(X_train, 1) >= 50 ||
        error("Too few training rows ($(size(X_train,1))). " *
              "Need at least 50 for meaningful GP evaluation.")

    # ── Validate config against actual feature count ──────────────────────────
    validate_config(GP_CFG)

    @info "Starting GP: $(GP_CFG.population_size) individuals × " *
          "$(GP_CFG.n_generations) generations × " *
          "$(GP_CFG.n_islands) islands"
    @info "Features: $n_features  |  Train rows: $(size(X_train,1))  " *
          "|  Test rows: $(size(X_test,1))"

    # ── Run GP ────────────────────────────────────────────────────────────────
    t_start = time()
    hof = run_gp(X_train, y_train, GP_CFG)
    t_elapsed = time() - t_start

    @info @sprintf("GP complete in %.1f seconds (%.2f min)", t_elapsed, t_elapsed/60)
    @info "HoF size: $(length(hof.members)) members"

    # ── OOS evaluation ────────────────────────────────────────────────────────
    indiv_df, tree_tables = evaluate_oos(hof, X_test, y_test)
    _print_oos_table(indiv_df)

    for (ind_rank, member_tree_df) in enumerate(tree_tables)
        _print_tree_oos_table(member_tree_df; max_rows = 13)
    end


    # ── Save results ──────────────────────────────────────────────────────────
    mkpath(RESULTS_DIR)
    
    summary_path = joinpath(RESULTS_DIR, "hof_summary.csv")
    CSV.write(summary_path, indiv_df)



    tree_dir = joinpath(RESULTS_DIR, "trees")
    mkpath(tree_dir)
    for (ind_rank, member_tree_df) in enumerate(tree_tables)
        CSV.write(joinpath(tree_dir, "individual_$(lpad(ind_rank, 2, '0'))_trees.csv"), member_tree_df)
    end


    @info "HoF summary saved to $summary_path"
    @info "Tree-level summary saved to $tree_dir"

    save_hof_signals(hof, X_test, RESULTS_DIR)

    # ── Print best formula ────────────────────────────────────────────────────
    if !isempty(hof.members)
        best_idx = argmax(m.pps_score for m in hof.members)
        best     = hof.members[best_idx]
        println("\n  Best HoF member (by train PPS):")
        for (k, tree) in enumerate(best.trees)
            println("    Tree $k: $(tree_to_string(tree))")
        end
        println()
    end

    return hof, indiv_df, tree_tables
end

# Run when invoked as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end