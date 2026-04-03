
# ====================================================================
# Types.jl — Core data structures for GP Nodes, trees, individuals.
# ====================================================================

#-------- Node type constants --------
const NODE_OPERATOR::Int8 = Int8(0)
const NODE_VARIABLE::Int8 = Int8(1)
const NODE_CONSTANT::Int8 = Int8(2)

## NODE (Flat, Type-Stable) 
# 
"""
    Node

A single node in a flat-array GP expression tree.

All fields are primitive integer types — `Node` is `isbits`, meaning Julia
stores it inline with zero heap allocation. A `Vector{Node}` is a single
contiguous allocation: cache-friendly, SIMD-friendly, trivially serializable
across `Distributed.jl` workers.

Fields
------
- `node_type`  : NODE_OPERATOR (0), NODE_VARIABLE (1), NODE_CONSTANT (2)
- `op_idx`     : 1-based index into OPERATOR_TABLE; 0 if not an operator node
- `left`       : index of left child in Tree.nodes[]; 0 = null (no child)
- `right`      : index of right child in Tree.nodes[]; 0 = null (unary or leaf)
- `feature`    : column index into X if variable node; 0 otherwise
- `const_idx`  : index into Tree.constants[] if constant node; 0 otherwise
"""

struct Node
    node_type::Int8   
    op_idx::Int8      
    left::Int16         
    right::Int16        
    feature::Int16     
    const_idx::Int16   
end


#-------- Constructor functions for different node types --------
"""Creates a binary operator node (left and right are non-zero child indices)."""
@inline op_node(op_idx::Int8, left::Int16, right::Int16) = Node(NODE_OPERATOR, op_idx, left, right, Int16(0), Int16(0))

"""Creates a unary operator node (right = 0 for 'no right child')."""
@inline unary_node(op_idx::Int8, left::Int16) = Node(NODE_OPERATOR, op_idx, left, Int16(0), Int16(0), Int16(0))

"""Creates a variable (feature) node."""
@inline variable_node(feature::Integer) = Node(NODE_VARIABLE, Int8(0), Int16(0), Int16(0), Int16(feature), Int16(0))

"""Creates a constant leaf node."""
@inline constant_node(const_idx::Integer) = Node(NODE_CONSTANT, Int8(0), Int16(0), Int16(0), Int16(0), Int16(const_idx))




#-----------------------------------------------------------
#-------- Node predicates --------
is_operator(n::Node) = n.node_type == NODE_OPERATOR
is_variable(n::Node) = n.node_type == NODE_VARIABLE
is_constant(n::Node) = n.node_type == NODE_CONSTANT
is_leaf(n::Node) = n.node_type != NODE_OPERATOR
is_unary_node(n::Node) = is_operator(n) && n.right == Int16(0) # true iff unary operator (one child)
is_binary_node(n::Node) = is_operator(n) && n.right != Int16(0) # true iff binary operator (two children)





#-----------------------------------------------------------
##  TREE STRUCTURE (Cache-Local, Flat Array)
#-----------------------------------------------------------
"""
    Tree

A complete expression tree stored as a flat array of Node structs.

Layout
------
nodes[1] is always the root. Left/right children are referenced by integer
index (node.left, node.right). The index 0 is the null sentinel.


Fields
------
- `nodes`     : all nodes; root is nodes[1]
- `constants` : Float64 values indexed by Node.const_idx; handed to Optim.jl
- `size`      : number of active nodes
- `depth`     : cached max depth — maintained by all mutating operations
"""
mutable struct Tree
    nodes::Vector{Node}    # Flat array of nodes: root is always nodes[1]
    constants::Vector{Float64}  # Scalar constants used by the tree
    size::Int       # Number of nodes currently in the tree
    depth::Int      # Depth of the tree
end


""" Empty Tree Initialization - zero nodes, zero constants"""
Tree() = Tree(Node[], Float64[], 0, 0)



##----------------------------------------------------------
## Mutation Probabilities
##----------------------------------------------------------
"""
    MutationProbs

Encapsulates the four mutation operator probabilities. The remaining
probability (1 - sum) is the chance of no mutation being applied.

Constructor: MutationProbs(subtree, point, hoist, shrink)
Validation:  sum must be ≤ 1.0; all values must be ≥ 0.
"""
struct MutationProbs
    subtree::Float64
    point::Float64
    hoist::Float64
    shrink::Float64
    """
        MutationProbs(subtree, point, hoist, shrink)
    
    Validates that probabilities are non-negative and sum to ≤ 1.0.
    """
    function MutationProbs(subtree::Float64, point::Float64, hoist::Float64, shrink::Float64)
        for (name, v) in (("subtree", subtree), ("point", point),
                          ("hoist", hoist), ("shrink", shrink))
            v >= 0.0 || error("MutationProbs: $name must be >= 0 (got $v)")
        end
        s = subtree + point + hoist + shrink
        s <= 1.0 || error("MutationProbs: probabilities must sum to <= 1.0 (got $s)")
        new(subtree, point, hoist, shrink)
    end
end




"""
    FitnessWeights

Encapsulates the four fitness component weights. Must sum to ≤ 1.0;
the remainder (1 - sum) is implicitly assigned to PPS as the primary signal.

Fields
──────
- `pps`     : weight for Predictive Power Score (IC + RankIC blend)
- `rre`     : weight for Relative Rank Entropy (signal concentration)
- `pfs`     : weight for Perturbation Fidelity Score (robustness to noise)
- `novelty` : weight for behavioral novelty (annealed over the run)

Constructor validates non-negativity and sum ≤ 1.0.
"""
struct FitnessWeights
    pps::Float64
    rre::Float64
    pfs::Float64
    novelty::Float64
    function FitnessWeights(pps::Float64, rre::Float64,
                             pfs::Float64, novelty::Float64)
        for (name, v) in (("pps", pps), ("rre", rre),
                          ("pfs", pfs), ("novelty", novelty))
            v >= 0.0 || error("FitnessWeights: $name must be >= 0 (got $v)")
        end
        s = pps + rre + pfs + novelty
        s <= 1.0 || error("FitnessWeights: weights must sum to <= 1.0 (got $s)")
        new(pps, rre, pfs, novelty)
    end
end






##----------------------------------------------------------
## Individual (Multi-Tree / Multi-Genome)
##----------------------------------------------------------
"""
    Individual

A GP individual composed of N expression trees whose outputs are combined
into a single factor signal.

Combination methods
-------------------
- :weighted_sum  →  Σ wᵢ · factorᵢ(X)        weights co-optimized with constants
- :rank_average  →  mean(rank(factorᵢ(X)))    robust; no weight tuning needed
- :vote          →  sign(Σ sign(factorᵢ(X))) directional majority; interpretable

Fields
------
- `trees`       : N expression trees
- `weights`     : combination weights; length = N
- `combination` : :weighted_sum | :rank_average | :vote
- `fitness`     : IC-based fitness (-Inf until first evaluation)
- `complexity`  : Σ tree.size — parsimony target for Pareto HoF
- `age`         : generation created — reserved for ALPS age-layering
- `behavior`    : output on fixed subsample rows — behavioral novelty fingerprint
- `dominated`   : true if dominated on the IC × complexity Pareto front
"""
mutable struct Individual
    trees::Vector{Tree}    
    weights::Vector{Float64}   
    combination::Symbol
    fitness::Float64        
    complexity::Int        
    age::Int                
    behavior::Vector{Float32}
    dominated::Bool        

    pps_score::Float64
    rre_score::Float64

end    

"""
    Individual(trees, weights, combination)

Primary constructor. Validates inputs, auto-derives complexity from tree sizes,
and enforces fitness = -Inf so unevaluated individuals never win selection.
"""
# Individual constructor with default values
function Individual(trees::Vector{Tree}, weights::Vector{Float64}, combination::Symbol)
    length(trees) == length(weights) ||
        error("Individual: trees and weights must have equal length " *
              "(got $(length(trees)) trees, $(length(weights)) weights)")
    combination in (:weighted_sum, :rank_average, :vote) ||
        error("Individual: unknown combination '$combination'. " *
              "Valid: :weighted_sum, :rank_average, :vote")
    complexity = sum(t.size for t in trees; init=0)
    Individual(trees, weights, combination, -Inf, complexity, 0, Float32[], false, 0.0, 0.0)
end











##----------------------------------------------------------
## CONFIGURATION
##----------------------------------------------------------
"""
    GPConfig

All GP hyperparameters in one struct with validated defaults.

Usage
-----
    cfg = GPConfig()                                     # all defaults
    cfg = GPConfig(population_size=10_000, n_islands=8) # override specific fields

Always call validate_config(cfg) before starting a run.
"""
Base.@kwdef struct GPConfig
    # Population
    population_size::Int = 1000
    n_trees_per_individual::Int = 10
    combination::Symbol = :rank_average  # :weighted_sum, :rank_average, :vote 

    # Constraints
    max_depth::Int = 6
    min_depth::Int = 2


    # Evolutionary Parameters
    n_generations::Int = 50
    elite_count::Int = 20
    crossover_prob::Float64 = 0.9
    mutation_prob::MutationProbs = MutationProbs(0.4, 0.35, 0.15, 0.1)

    # Selection & Optimization
    selection::Symbol = :epsilon_lexicase  # :tournament, :epsilon_lexicase
    tournament_k::Int = 5
    lexicase_n_periods::Int = 20
    lexicase_epsilon::Float64 = 0.1
    optimize_constants::Bool = true
    const_opt_method::Symbol = :bfgs  # :bfgs, :nelder_mead
    const_opt_every_n_gens::Int = 5


    # Fitness & Diversity
    parsimony::Float64 = 0.001
    fitness_metric::Symbol = :ic  
    fitness_weights::FitnessWeights = FitnessWeights(0.3, 0.3, 0.3, 0.1) # pps, rre, pfs, novelty       
    novelty_k::Int = 10
    novelty_archive_size::Int = 200
    behavior_sample_size::Int = 500
    hof_size::Int = 50
    pfs_noise_scale::Float64 = 0.01
    pfs_n_perturbations::Int = 5


    # Distribute / Island Model
    n_islands::Int = 4
    migration_interval::Int = 10
    migration_rate::Float64 = 0.05
    migration_topology::Symbol = :ring  # :ring, :star, :random

    # Operators -- Tier 1 
    unary_op_names::Vector{Symbol}  = [:safe_sqrt, :safe_log, :safe_inv, :safe_neg,
                                        :safe_sin,  :safe_cos, :safe_tan, :safe_tanh, :abs]
    binary_op_names::Vector{Symbol} = [:safe_add, :safe_sub, :safe_mul,
                                        :safe_div, :safe_pow, :signed_power]

    # Operators -- Tier 2 (Time-Series & Cross-Sectional)
    ts_op_names::Vector{Symbol}        = [:ts_delta, :ts_mean, :ts_stddev, :ts_rank,
                                           :ts_max,   :ts_min,  :ts_sum,   :decay]
    ts_binary_op_names::Vector{Symbol} = [:ts_corr, :ts_cov]
    cs_op_names::Vector{Symbol}        = [:cs_rank, :cs_zscore, :cs_scale, :cs_winsorize]

    # Window sizes sampled when generating TS nodes during tree building
    ts_window_sizes::Vector{Int} = [3, 5, 10, 20]
    
    const_prob::Float64 = 0.1
    fin_op_prob::Float64 = 0.5


    # Execution
    seed::Int = 42
    eval_subsample::Float64 = 0.5
    verbose::Bool = true

end



#----------------------------------------------------------
# Config Validation
#----------------------------------------------------------

"""
    validate_config(cfg::GPConfig)

Assert all invariants assumed by the rest of the codebase.
Call once at engine startup — far better to fail loudly here than
silently misbehave 80 generations into a 6-hour run.

Throws an AssertionError on the first violated constraint.
"""
function validate_config(cfg::GPConfig)

    # Tree shape
    @assert cfg.min_depth >= 1 "min_depth must be >= 1 (got $(cfg.min_depth))"
    @assert cfg.max_depth > cfg.min_depth "max_depth must be > min_depth (got max=$(cfg.max_depth), min=$(cfg.min_depth))"

    # Population
    @assert cfg.population_size > 0 "population_size must be > 0"
    @assert cfg.n_trees_per_individual >= 1 "n_trees_per_individual must be >= 1"
    @assert cfg.elite_count < cfg.population_size "elite_count ($(cfg.elite_count)) must be < population_size ($(cfg.population_size))"
    @assert cfg.n_islands >= 1 "n_islands must be >= 1"
    @assert cfg.population_size % cfg.n_islands == 0 "population_size must be divisible by n_islands ($(cfg.population_size) % $(cfg.n_islands) = $(cfg.population_size % cfg.n_islands))"

    # Mutation probabilities
    total_mut = cfg.mutation_prob.subtree + cfg.mutation_prob.point +
                cfg.mutation_prob.hoist   + cfg.mutation_prob.shrink
    @assert total_mut <= 1.0 "mutation_prob must sum to <= 1.0 (got $total_mut)"

    #Lexicase 
    @assert cfg.lexicase_n_periods >=2  "n_periods for lexicase method must be >= 2 (got $(cfg.lexicase_n_periods))"

    # Probabilities in (0,1]
    @assert 0.0 < cfg.crossover_prob <= 1.0 "crossover_prob must be in (0, 1] (got $(cfg.crossover_prob))"
    @assert 0.0 < cfg.eval_subsample <= 1.0 "eval_subsample must be in (0, 1] (got $(cfg.eval_subsample))"
    @assert 0.0 < cfg.migration_rate < 1.0 "migration_rate must be in (0, 1) (got $(cfg.migration_rate))"
    # fitness_weights validated by FitnessWeights constructor (non-negative, sum ≤ 1.0)
    @assert cfg.pfs_noise_scale > 0.0 "pfs_noise_scale must be > 0 (got $(cfg.pfs_noise_scale))"
    @assert cfg.pfs_n_perturbations >= 1 "pfs_n_perturbations must be >= 1 (got $(cfg.pfs_n_perturbations))"
    @assert 0.0 <= cfg.const_prob <= 1.0 "const_prob must be in [0, 1] (got $(cfg.const_prob))"
    @assert 0.0 <= cfg.fin_op_prob <= 1.0 "financial_op_prob must be in [0, 1] (got $(cfg.fin_op_prob))"
    @assert length(cfg.ts_window_sizes) >= 1 "ts_window_sizes must have at least one entry"
    @assert all(d >= 2 for d in cfg.ts_window_sizes) "all ts_window_sizes must be >= 2 (need at least 2 rows for rolling ops)"

    # Enum-like symbols
    @assert cfg.combination in (:weighted_sum, :rank_average, :vote) "unknown combination: '$(cfg.combination)'. Valid: :weighted_sum, :rank_average, :vote"
    @assert cfg.selection in (:epsilon_lexicase, :tournament) "unknown selection: '$(cfg.selection)'. Valid: :epsilon_lexicase, :tournament"
    @assert cfg.migration_topology in (:ring, :random, :star) "unknown migration_topology: '$(cfg.migration_topology)'. Valid: :ring, :random, :star"
    @assert cfg.const_opt_method in (:bfgs, :nelder_mead) "unknown const_opt_method: '$(cfg.const_opt_method)'. Valid: :bfgs, :nelder_mead"
    @assert cfg.fitness_metric in (:ic, :icir) "unknown fitness_metric: '$(cfg.fitness_metric)'. Valid: :ic, :icir"

    # Positive integers
    @assert cfg.tournament_k >= 2 "tournament_k must be >= 2 (got $(cfg.tournament_k))"
    @assert cfg.novelty_k >= 1 "novelty_k must be >= 1"
    @assert cfg.novelty_archive_size >= 1 "novelty_archive_size must be >= 1"
    @assert cfg.behavior_sample_size >= 10 "behavior_sample_size must be >= 10"
    @assert cfg.hof_size >= 1 "hof_size must be >= 1"
    @assert cfg.migration_interval >= 1 "migration_interval must be >= 1"
    @assert cfg.const_opt_every_n_gens >= 1 "const_opt_every_n_gens must be >= 1"

    nothing     # returns nothing on success; throws AssertionError on any failure
end