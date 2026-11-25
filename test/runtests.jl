# Run:   julia --project=. test/runtests.jl
# (or in VS Code: open the file, "Julia: Execute File" — also repeatedly)

# Activate the project environment if not done yet (otherwise e.g. JuMP is missing).
import Pkg
const _ROOT = normpath(joinpath(@__DIR__, ".."))
Base.active_project() == joinpath(_ROOT, "Project.toml") || Pkg.activate(_ROOT; io = devnull)

# Own module: when re-run in the same REPL it is replaced completely.
# Without it JSSPLab would be redefined in Main and all exports (read_instance, ...)
# would be ambiguous -> UndefVarError.
module JSSPLabTests

using Test
include(joinpath(@__DIR__, "..", "src", "JSSPLab.jl"))
using .JSSPLab

datadir = joinpath(@__DIR__, "..", "data", "instances")

@testset "JSSPLab" begin
    @testset "Reading" begin
        inst = read_instance(joinpath(datadir, "example3x3.txt"))
        @test inst.n == 3 && inst.m == 3
        @test inst.machine[1, :] == [3, 1, 2]     # Job 1: M3 → M1 → M2
        @test inst.ptime[1, :] == [2, 1, 2]

        ft06 = read_instance(joinpath(datadir, "ft06.txt"))
        @test ft06.n == 6 && ft06.m == 6
        @test trivial_lower_bound(ft06) <= 55      # optimum of ft06 is 55
    end

    @testset "Start solution & checker" begin
        for f in ["example3x3.txt", "ft06.txt"]
            inst = read_instance(joinpath(datadir, f))
            s = initial_schedule(inst)
            ok, _ = check_schedule(inst, s; verbose = false)
            @test ok
            @test makespan(inst, s) >= trivial_lower_bound(inst)
        end
    end

    @testset "Checker detects errors" begin
        inst = read_instance(joinpath(datadir, "example3x3.txt"))
        s = initial_schedule(inst)
        bad = Schedule(copy(s.start))
        bad.start[1, 2] = 0                         # violates job order
        ok, errs = check_schedule(inst, bad; verbose = false)
        @test !ok && !isempty(errs)

        ok2, _ = check_schedule(inst, s; claimed_makespan = makespan(inst, s) - 1, verbose = false)
        @test !ok2                                   # wrong makespan is detected
    end

    @testset "JSPLIB" begin
        names = list_jsplib()
        @test length(names) == 162
        for name in names                            # all files readable & consistent
            inst = load_jsplib(name)
            info = jsplib_info(name)
            @test (inst.n, inst.m) == (info.jobs, info.machines)
            info.upper === nothing || @test trivial_lower_bound(inst) <= info.upper   # ta71–80: no values
        end
        @test jsplib_info("ft06").optimum == 55
        @test load_jsplib("ft06").ptime == read_instance(joinpath(datadir, "ft06.txt")).ptime
        @test jsplib_info("abz8").optimum === nothing && jsplib_info("abz8").lower == 645
        for name in list_jsplib(max_jobs = 10, max_machines = 10)
            inst = load_jsplib(name)
            @test check_schedule(inst, initial_schedule(inst); verbose = false)[1]
        end
    end

    @testset "MIP model (Gurobi)" begin
        for (inst, opt) in [(read_instance(joinpath(datadir, "ft06.txt")), 55),
                            (load_jsplib("la01"), 666)]
            for ws in (nothing, initial_schedule(inst))
                res = solve_model(inst; time_limit = 60, silent = true, warm_start = ws)
                @test res.incumbent == opt && res.bound == opt
                @test check_schedule(inst, res.schedule; claimed_makespan = res.incumbent,
                                     verbose = false)[1]
            end
            # Big-M from the initial makespan: same optimum, start solution stays valid
            s0 = initial_schedule(inst)
            res = solve_model(inst; time_limit = 60, silent = true, warm_start = s0,
                              ub = makespan(inst, s0))
            @test res.incumbent == opt && res.bound == opt
        end
    end

    @testset "Random Fix-and-Optimize" begin
        inst = read_instance(joinpath(datadir, "ft06.txt"))
        start_ms = makespan(inst, initial_schedule(inst))
        n_cap = floor(Int, FREE_CAP * length(machine_pairs(inst)))   # ft06: 0.75 × 90 = 67

        # Feasible solutions ≤ start solution, for small and capped subproblems (fixed n)
        for nf in (10, 40, 1000)
            res = random_fao(inst; time_limit = 3, n_free_min = nf, n_free_max = nf)
            @test check_schedule(inst, res.schedule; claimed_makespan = res.incumbent,
                                 verbose = false)[1]
            @test res.incumbent <= start_ms
            @test res.trace[1].incumbent == start_ms          # first row = start solution
            @test issorted([tp.time for tp in res.trace])
            @test !isempty(res.iterations)
            @test res.params.seed == 42
            @test all(it -> it.n_free_y == min(nf, n_cap), res.iterations)   # exactly n
        end

        # Range: n drawn from n_free_min:n_free_max, Big-M = initial makespan
        res = random_fao(inst; time_limit = 3, n_free_min = 20, n_free_max = 50, ub = start_ms)
        @test all(it -> 20 <= it.n_free_y <= 50, res.iterations)
        @test length(unique(it.n_free_y for it in res.iterations)) > 1
        @test res.incumbent <= start_ms
        @test check_schedule(inst, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]

        # random_free returns exactly n_free_y distinct machine pairs
        ft10 = load_jsplib("ft10")
        all_pairs = Set(machine_pairs(ft10))
        for nf in (1, 37, 200, 450)
            sel = random_free(ft10, initial_schedule(ft10), [], JSSPLab.MersenneTwister(42), nf)
            @test length(sel) == length(unique(sel)) == nf
            @test all(in(all_pairs), sel)
        end

        # Static bound, gap, status
        res = random_fao(inst; time_limit = 3)
        @test res.bound == lower_bound(inst) == 52             # one-machine bound
        @test res.gap ≈ (res.incumbent - 52) / res.incumbent
        @test res.status == (res.incumbent == 52 ? "OPTIMAL" : "TIME_LIMIT")

        # accept_equal: no spurious improvements in the anytime trace
        r = random_fao(ft10; time_limit = 5, accept_equal = true)
        incs = [tp.incumbent for tp in r.trace]
        @test all(diff(incs[1:end-1]) .< 0)                    # only real improvements ...
        @test incs[end] == incs[end-1]                         # ... plus final row
        @test count(it -> it.improved, r.iterations) == length(r.trace) - 2
        @test all(it -> !it.improved || it.accepted, r.iterations)
        @test count(it -> it.accepted, r.iterations) >= count(it -> it.improved, r.iterations)

        # Reproducibility: same seed -> same freed y (first 10 iterations)
        function recorded_run(seed)
            log = Vector{Vector{Tuple{Tuple{Int,Int},Tuple{Int,Int}}}}()
            sel = (inst, cur, hist, rng, n) -> begin
                p = random_free(inst, cur, hist, rng, n)
                length(log) < 10 && push!(log, sort(p))
                p
            end
            random_fao(ft10; time_limit = 2, n_free_min = 100, n_free_max = 100, seed = seed,
                       select_free = sel)
            return log
        end
        log1, log2, log3 = recorded_run(42), recorded_run(42), recorded_run(43)
        @test length(log1) == 10 && log1 == log2
        @test log1 != log3                                     # other seed, other neighbourhoods

        # Hard time limit: swv12 (50×10) with 10 s ends after ≤ 11 s wall-clock time
        swv12 = load_jsplib("swv12")
        t = time()
        r = random_fao(swv12; time_limit = 10)
        @test time() - t <= 11
        @test check_schedule(swv12, r.schedule; claimed_makespan = r.incumbent, verbose = false)[1]

        # Bound = optimum (la01): FaO proves optimality and stops early
        la01 = load_jsplib("la01")
        r2 = random_fao(la01; time_limit = 30)
        @test r2.incumbent >= 666
        r2.incumbent == 666 && @test r2.status == "OPTIMAL" && r2.runtime < 30
    end

    @testset "Reference neighbourhood window_free" begin
        ft06 = read_instance(joinpath(datadir, "ft06.txt"))
        for inst in (ft06, load_jsplib("ft10"))
            all_pairs = Set(machine_pairs(inst))
            s = initial_schedule(inst)
            for nf in (1, 37, 200, 10_000)
                sel = window_free(inst, s, [], JSSPLab.MersenneTwister(42), nf)
                @test length(sel) == length(unique(sel)) == min(nf, length(all_pairs))
                @test all(in(all_pairs), sel)
                @test sel == window_free(inst, s, [], JSSPLab.MersenneTwister(42), nf)  # same seed
            end
        end
        res = random_fao(ft06; time_limit = 5, n_free_min = 40, n_free_max = 40,
                         select_free = window_free)
        @test check_schedule(ft06, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        @test res.incumbent <= makespan(ft06, initial_schedule(ft06))
    end

    @testset "Policy B (structural neighbourhoods)" begin
        Op = Tuple{Int,Int}
        for inst in (read_instance(joinpath(datadir, "ft06.txt")), load_jsplib("ft10"))
            s = initial_schedule(inst)
            F = compute_structural_features(inst, s)
            p = inst.ptime
            ops = Set(operations(inst))

            # slack ≥ 0; a critical path from time 0 whose length equals Cmax
            @test all(F.slack .>= 0)
            @test F.Cmax <= makespan(inst, s)
            msucc = Dict(seq[t] => seq[t+1] for seq in F.seqs for t in 1:length(seq)-1)
            succ(o) = [x for x in ((o[2] < inst.m ? (o[1], o[2] + 1) : nothing), get(msucc, o, nothing))
                       if x !== nothing]
            o = first(o for o in sort(collect(ops)) if F.critical[o...] && F.r[o...] == 0)
            len = p[o...]
            while !isempty(succ(o)) && F.q[o...] > 0
                nxt = first(x for x in succ(o) if p[x...] + F.q[x...] == F.q[o...])
                @test F.critical[nxt...] && F.r[nxt...] == F.r[o...] + p[o...]
                o = nxt
                len += p[o...]
            end
            @test len == F.Cmax
            @test all(b -> length(b) >= 2 && all(x -> F.critical[x...], b), F.blocks)

            # every type returns exactly k distinct valid operations, sorted, deterministic
            no_tabu = Set{Vector{Op}}()
            for k in (1, 13, length(ops)), ti in 1:4
                typ, _, R = JSSPLab.policyB_select(inst, F, k, ti, no_tabu)
                @test typ == JSSPLab.POLICY_B_TYPES[ti]
                @test length(R) == k && allunique(R) && all(in(ops), R) && issorted(R)
                @test R == JSSPLab.policyB_select(inst, F, k, ti, no_tabu)[3]
            end

            # tabu: a tabu set is never returned while non-tabu alternatives exist
            tabu = Set{Vector{Op}}()
            first_R = JSSPLab.policyB_select(inst, F, 13, 1, tabu)[3]
            for _ in 1:60
                typ, _, R = JSSPLab.policyB_select(inst, F, 13, 1, tabu)
                typ == :fallback && break
                @test !(R in tabu)
                push!(tabu, R)
            end
            @test first_R in tabu && length(tabu) > 1

            # fill_budget: k grows from policyA_k until the y budget n is (almost) used,
            # never above n, never fewer operations than the fixed-k variant
            pairs = machine_pairs(inst)
            nfree(R) = count(((a, b),) -> a in R || b in R, pairs)
            for n in (60, 200, 337), ti in 1:4
                k0 = JSSPLab.policyA_k(inst, n)
                _, _, R0 = JSSPLab.policyB_select(inst, F, k0, ti, no_tabu)
                _, _, R = JSSPLab.policyB_select(inst, F, k0, ti, no_tabu; n_free_y = n)
                @test nfree(Set(R)) <= n && length(R) >= length(R0) && allunique(R) && issorted(R)
                @test nfree(Set(R)) >= nfree(Set(R0))
            end
        end

        # PolicyB_v2: border logic on hand-made blocks (lengths 2, 3, 4, 7)
        blk(L) = [(j, 1) for j in 1:L]
        @test JSSPLab._border_ops(blk(2)) == blk(2)
        @test JSSPLab._border_ops(blk(3)) == blk(3)
        @test JSSPLab._border_ops(blk(4)) == blk(4)
        @test JSSPLab._border_ops(blk(7)) == [(1, 1), (2, 1), (6, 1), (7, 1)]

        for inst in (read_instance(joinpath(datadir, "ft06.txt")), load_jsplib("ft10"))
            F = compute_structural_features(inst, initial_schedule(inst))
            p = inst.ptime
            B = length(F.blocks)
            ranked = F.blocks[sort(1:B; by = i -> (-sum(p[o...] for o in F.blocks[i]), i))]
            # candidates: exactly B distinct ones, candidate i starts with block i's borders
            cands = JSSPLab.bb_candidates(inst, F, 10)
            @test length(cands) == B && allunique(first.(cands))
            for i in 1:B
                bi = JSSPLab._border_ops(ranked[i])
                @test cands[i][2](10)[1:length(bi)] == bi && cands[i][3] == length(cands[i][2](10))
            end
            # rotation for iterations 1..12 without tabu entries
            types = [JSSPLab.policyB_select(inst, F, 10, mod(it - 1, 6) + 1, Set{Vector{Op}}();
                                            n_free_y = 60, cycle = JSSPLab.POLICY_B_V2_CYCLE)[1]
                     for it in 1:12]
            @test types == repeat([:BB, :M, :BB, :TW, :BB, :J], 2)
        end

        # v2 in the loop: feasible, deterministic, never above the budget
        ft06v = read_instance(joinpath(datadir, "ft06.txt"))
        v2runs = map(1:2) do _
            st = policyB_v2_selector()
            res = random_fao(ft06v; time_limit = 3, n_free_min = 50, n_free_max = 50,
                             ub = makespan(ft06v, initial_schedule(ft06v)), select_free = st)
            res, policyB_log(st, res.iterations)
        end
        (rv, lv), (_, lv2) = v2runs
        @test check_schedule(ft06v, rv.schedule; claimed_makespan = rv.incumbent, verbose = false)[1]
        @test [r.type for r in lv[1:6]] == ["BB", "M", "BB", "TW", "BB", "J"] ||
              any(r -> r.n_tabu > 0, lv[1:6])            # tabu may divert after a non-improvement
        @test all(r -> r.n_free_y <= r.budget, lv)
        mv = min(length(lv), length(lv2))
        @test [(r.type, r.candidate, r.n_free_y) for r in lv[1:mv]] ==
              [(r.type, r.candidate, r.n_free_y) for r in lv2[1:mv]]

        # PolicyB_v3: diverse BB candidates (ft10 start solution has 3 critical blocks)
        ft10 = load_jsplib("ft10")
        s10 = initial_schedule(ft10)
        F = compute_structural_features(ft10, s10)
        B = length(F.blocks)
        @test B >= 2
        n = 337
        k0 = JSSPLab.policyA_k(ft10, n)
        slack_order = sort(operations(ft10); by = o -> (F.slack[o...], o))
        pairs_of = Dict(o => Tuple{Op,Op}[] for o in operations(ft10))
        for (a, b) in machine_pairs(ft10)
            push!(pairs_of[a], (a, b)); push!(pairs_of[b], (a, b))
        end
        rel(c) = JSSPLab._release_set(c[2], k0, F, slack_order, n, pairs_of)
        c3 = JSSPLab.bb_candidates(ft10, F, k0; n_free_y = n, pairs_of, single_focus = true)
        c2 = JSSPLab.bb_candidates(ft10, F, k0; n_free_y = n, pairs_of)
        @test length(c3) == B + 1 && [c[4] for c in c3] == 0:B
        sets3 = rel.(c3)
        # 1. after duplicate removal (first occurrence kept) the sets are pairwise distinct;
        #    here block 1 covers all of machine 8, so its focus set equals candidate 0's
        distinct_idx = [c[4] for (j, c) in enumerate(c3) if !(sets3[j] in sets3[1:j-1])]
        @test length(distinct_idx) >= 2 && allunique(sets3[distinct_idx .+ 1])
        @test rel(c3[1]) == rel(c2[1])                                # 3. candidate 0 == v2
        p10 = ft10.ptime
        ranked = F.blocks[sort(1:B; by = i -> (-sum(p10[o...] for o in F.blocks[i]), i))]
        for i in 1:B                                                   # 2. candidate i content
            b = ranked[i]; bo = JSSPLab._border_ops(b)
            seq = F.seqs[ft10.machine[b[1]...]]
            s = findfirst(==(b[1]), seq)
            raw = c3[i+1][2](k0)
            @test raw[1:length(bo)] == bo && c3[i+1][3] == length(bo)
            @test raw[length(bo)+1:end] ==
                  JSSPLab._expansion_order(seq, s, s + length(b) - 1)[1:length(raw)-length(bo)]
            @test issubset(bo, rel(c3[i+1]))
        end
        # 4. tabu progression with fixed incumbent and no improvements
        st = policyB_v3_selector()
        hist = NamedTuple[]
        for it in 0:6 * (B + 3) - 1
            st(ft10, s10, hist, nothing, n)
            push!(hist, (improved = false,))
        end
        bb_slots = [r for r in st.log if JSSPLab.POLICY_B_V2_CYCLE[mod(r.iteration - 1, 6) + 1] == :BB]
        nd = length(distinct_idx)
        @test [r.bb_candidate_index for r in bb_slots[1:nd]] == distinct_idx
        @test all(r -> r.type != "BB", bb_slots[nd+1:end])

        # v3 in the loop: feasible, deterministic, never above the budget
        v3runs = map(1:2) do _
            st = policyB_v3_selector()
            res = random_fao(ft10; time_limit = 4, ub = makespan(ft10, s10), select_free = st)
            res, policyB_log(st, res.iterations)
        end
        (r3, l3), (_, l3b) = v3runs
        @test check_schedule(ft10, r3.schedule; claimed_makespan = r3.incumbent, verbose = false)[1]
        @test all(r -> r.n_free_y <= r.budget, l3)
        m3 = min(length(l3), length(l3b))
        @test [(r.type, r.candidate, r.n_free_y) for r in l3[1:m3]] ==
              [(r.type, r.candidate, r.n_free_y) for r in l3b[1:m3]]

        # in the loop: feasible, ≤ start, round-robin CB → M → J → TW, deterministic
        ft06 = read_instance(joinpath(datadir, "ft06.txt"))
        runs = map(1:2) do _
            st = policyB_selector()
            res = random_fao(ft06; time_limit = 5, n_free_min = 50, n_free_max = 50,
                             ub = makespan(ft06, initial_schedule(ft06)), select_free = st)
            res, policyB_log(st, res.iterations)
        end
        (res, log), (_, log2) = runs
        @test check_schedule(ft06, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        @test res.incumbent <= makespan(ft06, initial_schedule(ft06))
        @test [r.type for r in log[1:4]] == ["CB", "M", "J", "TW"]
        @test all(r -> r.n_released == r.k, log)
        m = min(length(log), length(log2))
        @test [(r.type, r.candidate) for r in log[1:m]] == [(r.type, r.candidate) for r in log2[1:m]]
    end

    @testset "Policy C (adaptive deterministic control)" begin
        ft10 = load_jsplib("ft10")
        s0 = initial_schedule(ft10)
        runs = map(1:2) do _
            st = policyC_selector()
            feasible = Ref(true)
            f = (i, c, h, r, n) -> (feasible[] &= check_schedule(i, c; verbose = false)[1];
                                    st(i, c, h, r, n))
            res = random_fao(ft10; time_limit = 4, ub = makespan(ft10, s0), select_free = f)
            res, policyC_log(st, res.iterations), feasible[]
        end
        (res, log, feas), (_, log2, _) = runs
        @test feas                                                  # every incumbent seen
        @test check_schedule(ft10, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        @test [r.type for r in log[1:3]] == ["CB", "TW", "MA"] && all(r -> r.k == r.k_min, log[1:3])
        @test (log[1].k_min, log[1].k_max) == (22, 37)              # policyA_k(200), policyA_k(337)
        @test all(r -> r.k_min <= r.n_released == r.k <= r.k_max, log)
        @test all(r -> r.s_CB >= 0 && r.s_TW >= 0 && r.s_MA >= 0, log)
        m = min(length(log), length(log2))
        @test [(r.type, r.k, r.released) for r in log[1:m]] == [(r.type, r.k, r.released) for r in log2[1:m]]
        # size rule: after a failed, optimally solved iteration (t ≥ 3) k grows by Δ (unless restart/k_max)
        Δ = JSSPLab.policyC_delta(22, 37)
        for i in 3:length(log)-1
            a, b = log[i], log[i+1]
            (a.improved === false && a.sub_status == "OPTIMAL" && !b.restart) || continue
            @test b.k == min(a.k_max, a.k + Δ)
        end
        # neighbourhoods return exactly k distinct operations
        F = compute_structural_features(ft10, s0)
        for k in (1, 22, 37, 100), c in (1, 5)
            for R in (JSSPLab.policyC_cb(ft10, s0, F, k, c)[1], JSSPLab.policyC_tw(ft10, s0, k, c),
                      JSSPLab.policyC_ma(ft10, s0, k, c))
                @test length(R) == k && allunique(R)
            end
        end
    end

    @testset "Policy C v2" begin
        ft10 = load_jsplib("ft10")
        s0 = initial_schedule(ft10)
        runs = map(1:2) do _
            st = policyC_v2_selector()
            feasible = Ref(true)
            f = (i, c, h, r, n) -> (feasible[] &= check_schedule(i, c; verbose = false)[1];
                                    st(i, c, h, r, n))
            res = random_fao(ft10; time_limit = 4, ub = makespan(ft10, s0), select_free = f)
            res, policyC_v2_log(st, res.iterations), feasible[]
        end
        (res, log, feas), (_, log2, _) = runs
        @test feas
        @test check_schedule(ft10, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        @test [r.type for r in log[1:3]] == ["CB", "TW", "MA"] && all(r -> r.y_target == 200, log[1:3])
        @test (log[1].y_min, log[1].y_max) == (200, 337)
        solved = [r for r in log if !r.skipped && r.stop_reason === missing]
        @test all(r -> r.y_min <= r.n_free_y <= r.y_max, solved)
        # no set solved twice for the same incumbent makespan
        seen = Set{Tuple{Int,String}}()
        for r in solved
            @test !((r.Z_inc, r.released) in seen)
            push!(seen, (r.Z_inc, r.released))
        end
        # skipped iterations: logged as such, no Gurobi call, excluded from the diagnostics
        @test count(r -> isequal(r.sub_status, "SKIPPED_DUPLICATE"), log) == res.diagnostics.n_skipped
        @test res.diagnostics.iterations == count(it -> it.sub_status != "SKIPPED_DUPLICATE", res.iterations)
        # restarts only after F failures at y_max; scores never reset to 0
        @test all(i -> !log[i].restart || log[i-1].y_target == log[i].y_max, 2:length(log))
        @test all(i -> all(s -> !(getfield(log[i-1], s) > 0 && getfield(log[i], s) == 0),
                           (:s_CB, :s_TW, :s_MA)), 2:length(log))
        m = min(length(log), length(log2))
        @test [(r.type, r.y_target, r.released) for r in log[1:m]] ==
              [(r.type, r.y_target, r.released) for r in log2[1:m]]
        # STOP_SEARCH: trace continued to the time limit
        res.stop_reason == "exhausted" && @test res.trace[end].time == 4.0
        # LRU tie-break
        sc = Dict(:CB => 0.0, :TW => 0.0, :MA => 0.0)
        @test JSSPLab._policyC2_switch(:CB, sc, Dict(:CB => 9, :TW => 5, :MA => 3)) == :MA
        @test JSSPLab._policyC2_switch(:CB, sc, Dict(:CB => 9, :TW => 3, :MA => 3)) == :TW
        @test JSSPLab._policyC2_switch(:CB, Dict(:CB => 0.0, :TW => 0.1, :MA => 0.0),
                                       Dict(:CB => 9, :TW => 5, :MA => 3)) == :TW
    end

    @testset "Policy C v3" begin
        # Change A: switch by score only above ε = 1e-4, else least recently used, else priority
        lu = Dict(:CB => 9, :TW => 5, :MA => 3)
        sw(s, l = lu) = JSSPLab._policyC3_switch(:CB, Dict(:CB => 0.0, :TW => s[1], :MA => s[2]), l)
        @test sw((3e-4, 1e-4)) == (:TW, "score")
        @test sw((1e-5, 1e-3)) == (:MA, "score")
        @test sw((1.26e-12, 0.0)) == (:MA, "lru")          # v2 case: tiny scores carry no information
        @test sw((5e-5, 9e-5)) == (:MA, "lru")
        @test sw((2e-4, 2e-4)) == (:MA, "lru")             # tie above ε → LRU
        @test sw((0.0, 0.0), Dict(:CB => 9, :TW => 3, :MA => 3)) == (:TW, "priority")

        # Change B: fingerprint = exact machine sequences
        ft06 = read_instance(joinpath(datadir, "ft06.txt"))
        s6 = initial_schedule(ft06)
        fp = schedule_fingerprint(ft06, s6)
        @test length(fp) == ft06.m && sort(reduce(vcat, fp)) == collect(1:ft06.n * ft06.m)
        @test fp == schedule_fingerprint(ft06, schedule_from_sequences(ft06, machine_sequences(ft06, s6)))

        # in the loop: no (schedule, set) solved twice, deterministic, feasible, sizes
        ft10 = load_jsplib("ft10")
        s0 = initial_schedule(ft10)
        runs = map(1:2) do _
            st = policyC_v3_selector()
            sent = Any[]
            f = function (i, c, h, r, n)
                @test check_schedule(i, c; verbose = false)[1]
                out = st(i, c, h, r, n)
                out isa Vector && push!(sent, (machine_sequences(i, c), st.log[end].released))
                out
            end
            res = random_fao(ft10; time_limit = 4, ub = makespan(ft10, s0), select_free = f)
            res, policyC_v3_log(st, res.iterations), sent, st
        end
        (res, log, sent, st), (_, log2, _, _) = runs
        @test allunique(sent)                               # independent duplicate check
        @test check_schedule(ft10, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        solved = [r for r in log if !r.skipped && r.stop_reason === missing]
        @test all(r -> r.y_min <= r.n_free_y <= r.y_max, solved)
        @test allunique([(r.schedule_id, r.released) for r in solved])
        @test count(r -> r.sideways === true, log) == st.n_sideways
        @test maximum(r.schedule_id for r in log) == length(st.schedule_ids)
        @test any(r -> r.type == "TW", log[4:end])          # TW is used after warm-up
        m = min(length(log), length(log2))
        @test [(r.type, r.y_target, r.released, r.schedule_id) for r in log[1:m]] ==
              [(r.type, r.y_target, r.released, r.schedule_id) for r in log2[1:m]]
    end

    @testset "Policy D (hybrid)" begin
        # fill score on the 3×3 example (hand-checked values)
        ex = read_instance(joinpath(datadir, "example3x3.txt"))
        sx = initial_schedule(ex)
        Fx = compute_structural_features(ex, sx)
        cnt = Dict(o => 0 for o in operations(ex)); cnt[(1, 1)] = 2; cnt[(3, 1)] = 1
        fs = JSSPLab.policyD_fill_scores(ex, sx, Fx, cnt)
        @test fs[(3, 1)].score ≈ 0.30 * 1 + 0.25 * 1 + 0.15 * 1 + 0.30 * 0.5          # 0.85
        @test fs[(2, 1)].score ≈ 0.30 * (1 - 1 / 8) + 0 + 0.15 * 1 + 0.30 * 1          # 0.7125
        @test fs[(1, 2)].score ≈ 0.30 * 1 + 0 + 0.15 * 0.8 + 0.30 * 1                   # 0.72
        @test fs[(1, 1)].f_freq == 0.0 && fs[(2, 2)].f_wait ≈ 0.25
        @test all(v -> v.f_freq == 1.0, values(JSSPLab.policyD_fill_scores(ex, sx, Fx,
                                                Dict(o => 0 for o in operations(ex)))))

        # N ≤ 250 gate for Level 2
        @test JSSPLab.policyD_exact_allowed(load_jsplib("ft10"))     # 100 operations
        @test JSSPLab.policyD_exact_allowed(load_jsplib("la21"))     # 150
        @test JSSPLab.policyD_exact_allowed(load_jsplib("ta01"))     # 15 × 15 = 225
        @test !JSSPLab.policyD_exact_allowed(load_jsplib("swv12"))   # 500

        # exhaustion trigger, ESCAPE blocks, EXACT, leaving ESCAPE on a schedule change
        ft10 = load_jsplib("ft10")
        s0 = initial_schedule(ft10)
        fail = (improved = false, sub_status = "OPTIMAL", accepted = false)
        function drive!(st, s, hist, calls)
            out = nothing
            for _ in 1:calls
                out = st(ft10, s, hist, nothing, 300)
                push!(hist, fail)
                st.mode == :EXACT && break
            end
            out
        end
        st = policyD_selector()
        hist = NamedTuple[]
        drive!(st, s0, hist, 1)
        while st.mode == :STRUCT && length(hist) < 500         # fixed incumbent, no improvement
            drive!(st, s0, hist, 1)
        end
        @test st.mode == :ESCAPE && st.n_exhaustions == 1 && st.log[end].exhausted
        @test st.log[end].type == "RAND" && st.log[end].n_free_y == st.y_max
        @test any(r -> r.y_target == st.y_max, st.log)           # a full size cycle at y_max
        st_x = deepcopy(st); hist_x = copy(hist)
        out = drive!(st_x, s0, hist_x, 3 * 20 + 1)               # 3 blocks without change → EXACT
        @test st_x.mode == :EXACT && out === SOLVE_EXACT && st_x.escape_blocks == 3
        st_n = policyD_selector(variant = :noExact)              # noExact stays in ESCAPE
        hist_n = NamedTuple[]
        drive!(st_n, s0, hist_n, length(hist) + 3 * 20 + 5)
        @test st_n.mode == :ESCAPE && st_n.escape_blocks >= 3
        better = random_fao(ft10; time_limit = 2).schedule      # schedule change (improvement)
        @test makespan(ft10, better) < makespan(ft10, s0)
        st(ft10, better, hist, nothing, 300)
        @test st.mode == :STRUCT && st.y_target == st.y_min && st.escape_blocks == 0 &&
              st.fp_at_cycle_start == schedule_fingerprint(ft10, better)
        st_e = policyD_selector(variant = :noEscape)             # noEscape restarts instead
        hist_e = NamedTuple[]
        drive!(st_e, s0, hist_e, length(hist) + 10)
        @test st_e.mode == :STRUCT && st_e.n_exhaustions >= 1 && !any(r -> r.type == "RAND", st_e.log)

        # in the loop: deterministic, no duplicates, sizes, feasibility (incl. EXACT)
        runs = map(1:2) do _
            st = policyD_selector()
            sent = Any[]
            f = function (i, c, h, r, n)
                @test check_schedule(i, c; verbose = false)[1]
                out = st(i, c, h, r, n)
                out isa Vector && push!(sent, (machine_sequences(i, c), sort(out)))
                out
            end
            res = random_fao(ft10; time_limit = 6, ub = makespan(ft10, s0), select_free = f)
            res, policyD_log(st, res.iterations), sent
        end
        (res, log, sent), (_, log2, _) = runs
        @test allunique(sent)
        @test check_schedule(ft10, res.schedule; claimed_makespan = res.incumbent, verbose = false)[1]
        @test [r.type for r in log[1:6]] == ["BB", "M", "BB", "J", "BB", "TW"]
        @test all(r -> r.y_target == 200, log[1:6])
        solved = [r for r in log if !r.skipped && r.type in ("BB", "M", "J", "TW", "RAND")]
        @test all(r -> 200 <= r.n_free_y <= 337, solved)
        m = min(length(log), length(log2))
        @test [(r.mode, r.type, r.y_target, r.released, r.schedule_id) for r in log[1:m]] ==
              [(r.mode, r.type, r.y_target, r.released, r.schedule_id) for r in log2[1:m]]
    end

    @testset "Lower bounds" begin
        for name in list_jsplib(only_known_optimum = true)    # validity: LB ≤ optimum
            inst = load_jsplib(name)
            @test trivial_lower_bound(inst) <= lower_bound(inst) <= jsplib_info(name).optimum
        end
    end

    @testset "Schedule from sequences" begin
        inst = read_instance(joinpath(datadir, "ft06.txt"))
        s = initial_schedule(inst)
        s2 = schedule_from_sequences(inst, machine_sequences(inst, s))
        @test check_schedule(inst, s2; verbose = false)[1]
        @test makespan(inst, s2) <= makespan(inst, s)
    end
end

end # module JSSPLabTests
