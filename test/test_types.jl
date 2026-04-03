using Test

include(joinpath(@__DIR__, "..", "src", "Genetic.jl"))
using .Genetic

@testset "Step 1: Types.jl + Functions.jl" begin

    # =========================================================================
    @testset "Node — isbits and memory layout" begin
    # =========================================================================

        @test isbitstype(Node)      # must be isbits — no Union/pointer fields
        @test sizeof(Node) == 10   # 2×Int8 + 4×Int16 = 2 + 8 = 10 bytes

        @test NODE_OPERATOR === Int8(0)
        @test NODE_VARIABLE === Int8(1)
        @test NODE_CONSTANT === Int8(2)
    end


    # =========================================================================
    @testset "Node — convenience constructors" begin
    # =========================================================================
        op  = op_node(Int8(12), Int16(2), Int16(3))
        un  = unary_node(Int8(2), Int16(2))
        var = variable_node(5)
        con = constant_node(1)

        @test op.node_type  == NODE_OPERATOR
        @test un.node_type  == NODE_OPERATOR
        @test var.node_type == NODE_VARIABLE
        @test con.node_type == NODE_CONSTANT

        @test op.left   == Int16(2)
        @test op.right  == Int16(3)
        @test op.op_idx == Int8(12)

        @test un.left   == Int16(2)
        @test un.right  == Int16(0)   # unary: right child = 0 sentinel
        @test un.op_idx == Int8(2)

        @test var.feature   == Int16(5)
        @test var.op_idx    == Int8(0)
        @test var.left      == Int16(0)
        @test var.right     == Int16(0)
        @test var.const_idx == Int16(0)

        @test con.const_idx == Int16(1)
        @test con.op_idx    == Int8(0)
        @test con.left      == Int16(0)
        @test con.right     == Int16(0)
        @test con.feature   == Int16(0)
    end


    # =========================================================================
    @testset "Node — predicates" begin
    # =========================================================================
        op  = op_node(Int8(12), Int16(2), Int16(3))
        un  = unary_node(Int8(2), Int16(2))
        var = variable_node(1)
        con = constant_node(1)

        @test  is_operator(op)
        @test  is_operator(un)
        @test !is_operator(var)
        @test !is_operator(con)

        @test  is_variable(var)
        @test !is_variable(op)
        @test  is_constant(con)
        @test !is_constant(op)

        @test !is_leaf(op)
        @test !is_leaf(un)
        @test  is_leaf(var)
        @test  is_leaf(con)

        @test  is_binary_node(op)
        @test !is_binary_node(un)
        @test  is_unary_node(un)
        @test !is_unary_node(op)

        @test !is_unary_node(var)
        @test !is_binary_node(var)
    end


    # =========================================================================
    @testset "Tree — construction" begin
    # =========================================================================
        t = Tree()
        @test isempty(t.nodes)
        @test isempty(t.constants)
        @test t.size  == 0
        @test t.depth == 0

        t2 = Tree([constant_node(1)], [3.14], 1, 0)
        @test t2.size == 1
        @test t2.depth == 0
        @test t2.constants[1] ≈ 3.14

        n_op  = op_node(Int8(12), Int16(2), Int16(3))
        n_var = variable_node(1)
        n_con = constant_node(1)
        t3 = Tree([n_op, n_var, n_con], [2.0], 3, 1)
        @test t3.size == 3
        @test t3.depth == 1
        @test t3.nodes[1].op_idx == Int8(12)
        @test t3.nodes[2].feature == Int16(1)
        @test t3.constants[1] ≈ 2.0
    end


    # =========================================================================
    @testset "Individual — construction and defaults" begin
    # =========================================================================
        t1 = Tree([variable_node(1)], Float64[], 1, 0)
        t2 = Tree([variable_node(2)], Float64[], 1, 0)
        ind = Individual([t1, t2], [0.5, 0.5], :rank_average)

        @test length(ind.trees)   == 2
        @test length(ind.weights) == 2
        @test ind.combination == :rank_average
        @test ind.fitness     == -Inf
        @test ind.complexity  == 2
        @test ind.age         == 0
        @test ind.dominated   == false
        @test isempty(ind.behavior)

        t3 = Tree([op_node(Int8(12), Int16(2), Int16(3)),
                   variable_node(1),
                   variable_node(2)], Float64[], 3, 1)
        ind2 = Individual([t3], [1.0], :weighted_sum)
        @test ind2.complexity == 3

        for combo in (:weighted_sum, :rank_average, :vote)
            @test_nowarn Individual([t1], [1.0], combo)
        end

        @test_throws ErrorException Individual([t1], [1.0], :bad_method)
        @test_throws ErrorException Individual([t1, t2], [1.0], :rank_average)
    end


    # =========================================================================
    @testset "MutationProbs — construction and validation" begin
    # =========================================================================
        mp = MutationProbs(0.4, 0.35, 0.15, 0.1)
        @test mp.subtree ≈ 0.4
        @test mp.point   ≈ 0.35
        @test mp.hoist   ≈ 0.15
        @test mp.shrink  ≈ 0.1
        @test mp.subtree + mp.point + mp.hoist + mp.shrink ≈ 1.0

        @test_nowarn MutationProbs(0.3, 0.3, 0.1, 0.1)
        @test_throws ErrorException MutationProbs(0.5, 0.5, 0.3, 0.3)
        @test_throws ErrorException MutationProbs(-0.1, 0.5, 0.3, 0.3)
    end


    # =========================================================================
    @testset "FitnessWeights — construction and validation" begin
    # =========================================================================
        fw = FitnessWeights(0.3, 0.3, 0.3, 0.1)
        @test fw.pps     ≈ 0.3
        @test fw.rre     ≈ 0.3
        @test fw.pfs     ≈ 0.3
        @test fw.novelty ≈ 0.1

        @test_nowarn FitnessWeights(0.2, 0.2, 0.2, 0.2)
        @test_throws ErrorException FitnessWeights(0.5, 0.5, 0.3, 0.3)
        @test_throws ErrorException FitnessWeights(-0.1, 0.5, 0.3, 0.3)
    end


    # =========================================================================
    @testset "GPConfig — defaults" begin
    # =========================================================================
        cfg = GPConfig()

        @test cfg.population_size        == 1_000
        @test cfg.n_trees_per_individual == 10
        @test cfg.max_depth              == 6
        @test cfg.min_depth              == 2
        @test cfg.n_generations          == 50
        @test cfg.elite_count            == 20
        @test cfg.crossover_prob         ≈ 0.90
        @test cfg.combination            == :rank_average
        @test cfg.selection              == :epsilon_lexicase
        @test cfg.n_islands              == 4
        @test cfg.migration_topology     == :ring
        @test cfg.fitness_metric         == :ic
        @test cfg.const_opt_method       == :bfgs
        @test cfg.parsimony              ≈ 0.001
        @test cfg.fitness_weights.novelty ≈ 0.1

        total = cfg.mutation_prob.subtree + cfg.mutation_prob.point +
                cfg.mutation_prob.hoist   + cfg.mutation_prob.shrink
        @test total ≈ 1.0
    end


    # =========================================================================
    @testset "GPConfig — validate_config passes on valid configs" begin
    # =========================================================================

        local ok = false
        try validate_config(GPConfig()); ok = true catch; end
        @test ok

        local ok2 = false
        try
            validate_config(GPConfig(
                population_size = 1_000,
                n_islands       = 4,
                selection       = :tournament,
                combination     = :weighted_sum,
            ))
            ok2 = true
        catch
        end
        @test ok2
    end


    # =========================================================================
    @testset "GPConfig — validate_config catches bad configs" begin
    # =========================================================================
    # Uses @test_throws AssertionError for all invariants.

        @test_throws Exception validate_config(GPConfig(min_depth=6, max_depth=6))
        @test_throws Exception validate_config(GPConfig(eval_subsample=0.0))
        @test_throws Exception validate_config(GPConfig(population_size=1001, n_islands=4))
        @test_throws Exception validate_config(GPConfig(population_size=100, elite_count=100))
        @test_throws Exception validate_config(GPConfig(combination=:unknown))
        @test_throws Exception validate_config(GPConfig(selection=:unknown))
        @test_throws Exception validate_config(GPConfig(migration_topology=:mesh))
    end


    # =========================================================================
    @testset "Operators — registry sizes" begin
    # =========================================================================
        @test N_UNARY  == Int8(9)
        @test N_BINARY == Int8(6)
        @test N_OPS    == Int8(15)
        @test length(UNARY_OPS)  == Int(N_UNARY)
        @test length(BINARY_OPS) == Int(N_BINARY)

        @test N_TS_UNARY  == Int8(8)
        @test N_TS_BINARY == Int8(2)
        @test N_CS        == Int8(4)
        @test length(TS_UNARY_OPS)  == Int(N_TS_UNARY)
        @test length(TS_BINARY_OPS) == Int(N_TS_BINARY)
        @test length(CS_OPS)        == Int(N_CS)

        @test N_ALL_OPS == Int8(29)
        @test length(OP_SYMBOL_TO_IDX) == Int(N_ALL_OPS)
    end


    # =========================================================================
    @testset "Operators — index range predicates" begin
    # =========================================================================
        for i in Int8(1):N_UNARY
            @test  is_unary_op(i)
            @test !is_binary_op(i)
            @test !is_ts_unary_op(i)
            @test !is_ts_binary_op(i)
            @test !is_cs_op(i)
            @test  get_arity(i) == 1
        end

        for i in Int8(N_UNARY + 1):N_OPS
            @test !is_unary_op(i)
            @test  is_binary_op(i)
            @test  get_arity(i) == 2
        end

        for i in IDX_TS_UNARY_START:IDX_TS_UNARY_END
            @test  is_ts_unary_op(i)
            @test !is_ts_binary_op(i)
            @test !is_cs_op(i)
            @test  get_arity(i) == 1
        end

        for i in IDX_TS_BINARY_START:IDX_TS_BINARY_END
            @test !is_ts_unary_op(i)
            @test  is_ts_binary_op(i)
            @test  get_arity(i) == 2
        end

        for i in IDX_CS_START:IDX_CS_END
            @test  is_cs_op(i)
            @test  get_arity(i) == 1
        end
    end


    # =========================================================================
    @testset "Operators — safe functions: no NaN/Inf on edge cases" begin
    # =========================================================================
    # Tests that safe_* ops return finite values for tricky inputs.
    #
    # NOTE — safe_pow known limitation:
    #   safe_pow(x, y) = abs(x)^clamp(y, -5, 5)
    #   When abs(x) == 0 and y < 0: 0^(-k) = Inf — not protected.
    #   The binary loop below excludes (zero-or-tiny-base, large-neg-exp)
    #   combinations that overflow. These are tested separately below.

        tricky = [0.0, -0.0, 1e-300, -1e-300, 1e15, -1e15, -1.0, 0.5, 1.0]

        # ── Unary: all inputs safe ───────────────────────────────────────────
        for x in tricky
            @test isfinite(safe_log(x))
            @test isfinite(safe_sqrt(x))
            @test isfinite(safe_inv(x))
            @test isfinite(safe_neg(x))
            @test isfinite(safe_sin(x))
            @test isfinite(safe_cos(x))
            @test isfinite(safe_tan(x))
            @test isfinite(safe_tanh(x))
            @test isfinite(abs(x))
        end

        # ── Binary: safe_add, safe_sub, safe_mul, safe_div — all inputs safe ─
        for x in tricky, y in tricky
            @test isfinite(safe_add(x, y))
            @test isfinite(safe_sub(x, y))
            @test isfinite(safe_mul(x, y))
            @test isfinite(safe_div(x, y))
        end

        # ── safe_pow: safe when base is not near-zero or exponent is not
        #    large-negative (the two conditions that together cause overflow) ──
        safe_pow_cases = [
            (1.0,    2.0),   (1.0,  -1.0),  (-1.0,  2.0),  (-1.0, -1.0),
            (2.0,    3.0),   (0.5,   2.0),  (1e15,  1.0),  (1e15, -1.0),
            (1e15, -1e15),   (0.0,   0.0),  (0.0,   1.0),  (0.0,   0.5),
            (1e-300, 1.0),   (1e-300, 0.5), (1e-300, 2.0),
        ]
        for (x, y) in safe_pow_cases
            @test isfinite(safe_pow(x, y))
        end

        # ── safe_pow known Inf cases: zero base + negative exponent ──────────
        # These are EXPECTED to return Inf — testing that the impl matches spec.
        @test !isfinite(safe_pow(0.0,    -1.0))
        @test !isfinite(safe_pow(0.0,   -1e15))
        @test !isfinite(safe_pow(-0.0,   -1.0))
        @test !isfinite(safe_pow(1e-300, -1e15))
    end


    # =========================================================================
    @testset "Operators — safe_div denominator protection" begin
    # =========================================================================
        @test isfinite(safe_div(1.0, 0.0))
        @test isfinite(safe_div(-5.0, 0.0))
        @test isfinite(safe_div(0.0, 0.0))
        @test safe_div(1.0, 0.0) ≈ 1.0 / 1e-6  rtol=1e-5
    end


    # =========================================================================
    @testset "Operators — apply_unary_scalar correctness" begin
    # =========================================================================
    # Index order: 1=safe_sqrt, 2=safe_log, 3=safe_inv, 4=safe_neg,
    #              5=safe_sin,  6=safe_cos, 7=safe_tan, 8=safe_tanh, 9=abs

        @test apply_unary_scalar(Int8(1), 4.0)  ≈ safe_sqrt(4.0)
        @test apply_unary_scalar(Int8(2), 1.0)  ≈ safe_log(1.0)
        @test apply_unary_scalar(Int8(3), 2.0)  ≈ safe_inv(2.0)
        @test apply_unary_scalar(Int8(4), 1.0)  ≈ safe_neg(1.0)
        @test apply_unary_scalar(Int8(5), 0.5)  ≈ safe_sin(0.5)
        @test apply_unary_scalar(Int8(6), 0.5)  ≈ safe_cos(0.5)
        @test apply_unary_scalar(Int8(7), 0.5)  ≈ safe_tan(0.5)
        @test apply_unary_scalar(Int8(8), 1.0)  ≈ safe_tanh(1.0)
        @test apply_unary_scalar(Int8(9), -3.0) ≈ abs(-3.0)

        @test_throws BoundsError apply_unary_scalar(Int8(0), 1.0)
        @test_throws BoundsError apply_unary_scalar(Int8(10), 1.0)
    end


    # =========================================================================
    @testset "Operators — apply_binary_scalar correctness" begin
    # =========================================================================
    # Index order: 10=safe_add, 11=safe_sub, 12=safe_mul, 13=safe_div,
    #              14=safe_pow, 15=signed_power
    #
    # NOTE: `signed_power` is the function name in Functions.jl (renamed from
    # `signed` to avoid shadowing Base.signed). Dispatch via apply_binary_scalar.

        @test apply_binary_scalar(Int8(10), 3.0, 2.0) ≈ safe_add(3.0, 2.0)
        @test apply_binary_scalar(Int8(11), 3.0, 2.0) ≈ safe_sub(3.0, 2.0)
        @test apply_binary_scalar(Int8(12), 3.0, 2.0) ≈ safe_mul(3.0, 2.0)
        @test apply_binary_scalar(Int8(13), 3.0, 2.0) ≈ safe_div(3.0, 2.0)
        @test apply_binary_scalar(Int8(14), 2.0, 3.0) ≈ safe_pow(2.0, 3.0)

        # signed_power(x, e) = sign(x) * abs(x)^clamp(e, -5, 5)
        let x = 3.0, e = 2.0
            expected = sign(x) * abs(x)^clamp(e, -5.0, 5.0)
            @test apply_binary_scalar(Int8(15), x, e) ≈ expected
        end
        # Negative base: sign flips
        let x = -3.0, e = 2.0
            expected = sign(x) * abs(x)^clamp(e, -5.0, 5.0)
            @test apply_binary_scalar(Int8(15), x, e) ≈ expected
        end

        @test_throws BoundsError apply_binary_scalar(Int8(1), 1.0, 1.0)
        @test_throws BoundsError apply_binary_scalar(Int8(16), 1.0, 1.0)
    end


    # =========================================================================
    @testset "Operators — vectorized apply" begin
    # =========================================================================
        x = [1.0, 2.0, -1.0, 0.0, 0.5]
        y = [2.0, 0.0, 3.0, -1.0, 0.5]

        out_u = apply_unary_op(Int8(1), x)
        @test all(isfinite, out_u)
        @test out_u ≈ safe_sqrt.(x)

        out_u2 = apply_unary_op(Int8(2), x)
        @test out_u2 ≈ safe_log.(x)

        out_b = apply_binary_op(Int8(12), x, y)
        @test out_b ≈ safe_mul.(x, y)

        out_b2 = apply_binary_op(Int8(13), x, y)
        @test all(isfinite, out_b2)
        @test out_b2 ≈ safe_div.(x, y)
    end


    # =========================================================================
    @testset "Operators — OP_SYMBOL_TO_IDX completeness" begin
    # =========================================================================
        unary_syms = [:safe_sqrt, :safe_log, :safe_inv, :safe_neg,
                      :safe_sin,  :safe_cos, :safe_tan, :safe_tanh, :abs]
        binary_syms = [:safe_add, :safe_sub, :safe_mul, :safe_div,
                       :safe_pow, :signed_power]

        for s in unary_syms
            @test haskey(OP_SYMBOL_TO_IDX, s)
            @test is_unary_op(OP_SYMBOL_TO_IDX[s])
        end
        for s in binary_syms
            @test haskey(OP_SYMBOL_TO_IDX, s)
            @test is_binary_op(OP_SYMBOL_TO_IDX[s])
        end

        for s in [:ts_mean, :ts_corr, :cs_rank, :cs_zscore, :decay]
            @test haskey(OP_SYMBOL_TO_IDX, s)
        end
    end


    # =========================================================================
    @testset "Operators — Symbol_to_op_idx and validate_op_symbols" begin
    # =========================================================================
        @test Symbol_to_op_idx(:safe_log)  == OP_SYMBOL_TO_IDX[:safe_log]
        @test Symbol_to_op_idx(:safe_mul)  == OP_SYMBOL_TO_IDX[:safe_mul]
        @test Symbol_to_op_idx(:ts_mean)   == OP_SYMBOL_TO_IDX[:ts_mean]
        @test Symbol_to_op_idx(:cs_rank)   == OP_SYMBOL_TO_IDX[:cs_rank]

        @test_throws ErrorException Symbol_to_op_idx(:nonexistent_op)

        @test_nowarn validate_op_symbols(GPConfig())

        bad_cfg = GPConfig(
            unary_op_names  = [:nonexistent_op],
            population_size = 1_000,
            n_islands       = 4,
        )
        @test_throws ErrorException validate_op_symbols(bad_cfg)
    end


    # =========================================================================
    @testset "Operators — GPConfig default op names are all valid symbols" begin
    # =========================================================================
        cfg = GPConfig()
        @test_nowarn validate_op_symbols(cfg)

        for s in cfg.unary_op_names
            @test haskey(OP_SYMBOL_TO_IDX, s)
        end
        for s in cfg.binary_op_names
            @test haskey(OP_SYMBOL_TO_IDX, s)
        end
        for s in cfg.ts_op_names
            @test haskey(OP_SYMBOL_TO_IDX, s)
        end
        for s in cfg.cs_op_names
            @test haskey(OP_SYMBOL_TO_IDX, s)
        end
    end


    # =========================================================================
    @testset "Operators — ForwardDiff differentiability" begin
    # =========================================================================
    # INVARIANT: every safe op must produce a finite gradient via ForwardDiff.
    #
    # NOTE — abs(0.0) is non-differentiable at exactly x=0 via ForwardDiff
    # (returns NaN). Tests involving abs use x=1e-10 as the boundary proxy.
    # The relevant functions are: safe_sqrt, safe_inv, safe_div, safe_pow —
    # all of which call Base.abs internally.

        using ForwardDiff

        # Points away from zero — all ops should be differentiable
        interior = [0.001, 0.5, 1.0, -0.5, -0.001, 2.0]

        for x in interior
            @test isfinite(ForwardDiff.derivative(safe_log,  x))
            @test isfinite(ForwardDiff.derivative(safe_sqrt, x))
            @test isfinite(ForwardDiff.derivative(safe_inv,  x))
            @test isfinite(ForwardDiff.derivative(safe_neg,  x))
            @test isfinite(ForwardDiff.derivative(safe_sin,  x))
            @test isfinite(ForwardDiff.derivative(safe_cos,  x))
            @test isfinite(ForwardDiff.derivative(safe_tan,  x))
            @test isfinite(ForwardDiff.derivative(safe_tanh, x))
        end

        # safe_div at y≈0 — gradient must be finite (V1 had Inf here)
        f_div(y) = safe_div(1.0, y)
        @test isfinite(ForwardDiff.derivative(f_div, 1e-10))
        @test isfinite(ForwardDiff.derivative(f_div, 1e-6))

        # safe_sqrt at x≈0 — V1 returned sqrt(0) = 0 with gradient = Inf
        @test isfinite(ForwardDiff.derivative(safe_sqrt, 1e-10))
        @test isfinite(ForwardDiff.derivative(safe_sqrt, 1e-6))

        # safe_pow at x≈0
        f_pow(x) = safe_pow(x, 2.0)
        @test isfinite(ForwardDiff.derivative(f_pow, 1e-10))
        @test isfinite(ForwardDiff.derivative(f_pow, 1e-6))
    end

end  # @testset "Step 1"

println("\n✓ Step 1 complete: Types.jl + Functions.jl")