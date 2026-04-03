module Genetic

include("Types.jl")
include("Functions.jl")
include("Tree.jl")
include("Evaluate.jl")
include("Fitness.jl")
include("Init.jl")
include("Operators_GP.jl")
include("Constants.jl")
include("Selection.jl")
include("Diversity.jl")
include("Island.jl")
include("Engine.jl")


# ---------------------------------------------------------------------------
# Types.jl exports
# ---------------------------------------------------------------------------
export Node, Tree, Individual, GPConfig, MutationProbs, FitnessWeights
export NODE_OPERATOR, NODE_VARIABLE, NODE_CONSTANT
export op_node, unary_node, variable_node, constant_node
export is_operator, is_variable, is_constant, is_leaf
export is_unary_node, is_binary_node
export validate_config


# ---------------------------------------------------------------------------
# Functions.jl -- Tier 1: elementwise operators
# ---------------------------------------------------------------------------
export safe_log, safe_sqrt, safe_inv, safe_neg
export safe_sin, safe_cos, safe_tan, safe_tanh
export safe_add, safe_sub, safe_mul, safe_div, safe_pow, signed

# ---------------------------------------------------------------------------
# Functions.jl -- Tier 2: financial operators
# ---------------------------------------------------------------------------
export ts_delta, ts_mean, ts_stddev, ts_rank
export ts_max, ts_min, ts_sum, decay
export ts_corr, ts_cov
export cs_rank, cs_zscore, cs_scale, cs_winsorize

# ---------------------------------------------------------------------------
# Functions.jl -- Registry constants
# ---------------------------------------------------------------------------
export UNARY_OPS, BINARY_OPS, TS_UNARY_OPS, TS_BINARY_OPS, CS_OPS
export N_UNARY, N_BINARY, N_ELEM, N_OPS
export N_TS_UNARY, N_TS_BINARY, N_CS, N_ALL_OPS
export IDX_TS_UNARY_START,  IDX_TS_UNARY_END
export IDX_TS_BINARY_START, IDX_TS_BINARY_END
export IDX_CS_START,        IDX_CS_END
export OP_SYMBOL_TO_IDX, OP_NAMES

# ---------------------------------------------------------------------------
# Functions.jl -- Predicates & dispatch
# ---------------------------------------------------------------------------
export is_unary_op, is_binary_op, is_elem_op
export is_ts_unary_op, is_ts_binary_op, is_cs_op, is_financial_op
export get_arity, op_name
export apply_unary_scalar, apply_binary_scalar
export apply_unary_op, apply_binary_op
export apply_ts_unary_op, apply_ts_binary_op, apply_cs_op
export validate_op_symbols, Symbol_to_op_idx


# ---------------------------------------------------------------------------
# Tree.jl exports
# ---------------------------------------------------------------------------

# Op-set resolver
export make_op_sets

# Construction
export build_random_tree

# Evaluation
export eval_tree

# Copying
export copy_tree

# Structural utilities
export subtree_indices, subtree_size
export compute_depth, recompute_depth!
export replace_subtree

# Random sampling helpers (used by mutation / crossover)
export random_node_idx, random_nonterminal_idx

# Pretty-printing
export tree_to_string

# ---------------------------------------------------------------------------
# Evaluate.jl exports
# ---------------------------------------------------------------------------
export extract_constants, inject_constants!
export update_complexity!
export combine_trees
export update_behavior!

# ---------------------------------------------------------------------------
# Fitness.jl exports
# ---------------------------------------------------------------------------
export prerank_y
export ic, rank_ic, icir, rolling_rank_ic
export pps, rre, pfs
export current_novelty_weight
export compute_fitness, evaluate_fitness!
export evaluate_population!

# ---------------------------------------------------------------------------
# Init.jl exports
# ---------------------------------------------------------------------------
export init_population
export init_behavior_rows

#---------------------------------------------------------------------------
# Operators_GP.jl exports
#---------------------------------------------------------------------------
export crossover
export subtree_mutate, point_mutate, hoist_mutate, shrink_mutate
export apply_mutation

# ---------------------------------------------------------------------------
# Constants.jl exports
# ---------------------------------------------------------------------------
export eval_tree_with_consts
export optimize_constants!
export optimize_individual_constants!
export optimize_population_constants!

# ---------------------------------------------------------------------------
# Selection.jl exports
# ---------------------------------------------------------------------------
export tournament_select
export epsilon_lexicase_select
export select
export select_n
export build_error_matrix

# ---------------------------------------------------------------------------
# Diversity.jl exports
# ---------------------------------------------------------------------------
export NoveltyArchive, ParetoHoF
export behavioral_distance, knn_mean_distance
export compute_novelty_scores
export update_archive!
export pareto_dominates
export update_hof!
export hof_summary

# ---------------------------------------------------------------------------
# Island.jl exports
# ---------------------------------------------------------------------------
export partition_population, merge_islands
export migration_targets
export migrate!

# ---------------------------------------------------------------------------
# Engine.jl exports
# ---------------------------------------------------------------------------
export run_gp

end