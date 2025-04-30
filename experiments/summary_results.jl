using Printf
using Statistics

function read_results(fname::String)
    regex_results = r"\s*(\d+)\s+(\d+)\s+(\w+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+)\s+(\d+.\d+)"
    regex_no_bm = r"\s*(\d+)\s+(\d+)\s+(\w+)\s+-?(\d+.\d+e?[+-]?\d+)?\s+(\d+.\d+e?[+-]?\d+)"
    
    seeds = Int64[]
    iters = Int64[]
    statuss = Bool[]
    objs = Float64[]
    constrs = Float64[]
    walls = Float64[]
    solvers = Float64[]
    
    open(fname, "r") do io
        lines = readlines(io)
        
        for line in lines
            res = match(regex_results, line)
            if !isnothing(res)
                seed, iter, status, obj, constr, wall, solver = parse_results(res.captures)
                push!(seeds, seed)
                push!(iters, iter)
                push!(statuss, status)
                push!(objs, obj)
                push!(constrs, constr)
                push!(walls, wall)
                push!(solvers, solver)
            end
            !isnothing(res) && continue
            # no benchmarks run for wall/solver time
            res_no_bm = match(regex_no_bm, line)
            if !isnothing(res_no_bm)
                seed, iter, status, obj, constr, wall, solver = parse_results(res_no_bm.captures)
                push!(seeds, seed)
                push!(iters, iter)
                push!(statuss, status)
                push!(objs, obj)
                push!(constrs, constr)
                push!(walls, wall)
                push!(solvers, solver)
            end
        end
    end 
    
    return seeds, iters, statuss, objs, constrs, walls, solvers
end

function parse_results(res)
    seed = parse(Int64, res[1])
    iters = parse(Int64, res[2])
    status = parse(Bool, res[3])
    obj = parse(Float64, res[4])
    constr = parse(Float64, res[5])
    if length(res) == 7
        wall = parse(Float64, res[6])
        solver = parse(Float64, res[7])
    else
        wall = 0.0
        solver = 0.0
    end
    return seed, iters, status, obj, constr, wall, solver
end

using StatsPlots

function plot_results(problemclass)
    fnames_in = ["ipddp2/results/$problemclass.txt", "ipopt/results/$problemclass.txt", "proxddp/results/$problemclass.txt"]
    names = ["IPDDP2", "IPOPT", "ProxDDP"]

    if problemclass == "concar"
        ylims_time = (0, 1450)
        yticks_time = [0, 250, 500, 750, 1000, 1200]
        yticks_viol = [1e-16, 1e-8, 1e-4, 1e-2, 1]
        ylim_obs = (0.8, 2000)
        yticks_obs = [1, 1e1, 1e2, 1e3]
    elseif problemclass == "cartpole_friction"
        ylims_time = (50, 6000)
        yticks_time = [0, 1500, 3000, 4500, 6000]
        yticks_viol = [1e-16, 1e-8, 1e-4, 1e-2, 1]
        ylim_obs = (1e-3, 1e3)
        yticks_obs = [1e-1, 1e1, 1e3]
    end

    _, iters_ipd, _, objs_ipd, constrs_ipd, wall_ipd, _ = read_results("ipddp2/results/$problemclass.txt")
    _, iters_ipo, _, objs_ipo, constrs_ipo, wall_ipo, _ = read_results("ipopt/results/$problemclass.txt")
    _, iters_ipob, _, objs_ipob, constrs_ipob, wall_ipob, _ = read_results("ipopt/results/bfgs_$problemclass.txt")
    _, iters_al, _, objs_al, constrs_al, _, _ = read_results("proxddp/results/$problemclass.txt")

    boxplot(["IPDDP2" "IPOPT" "IPOPT (B)" "ProxDDP"], [objs_ipd objs_ipo objs_ipob objs_al], title="Objective Function",
                yaxis=:log10, legend=false, size=(300, 300))
    savefig("plots/$problemclass/objective.pdf")

    boxplot(["IPDDP2" "IPOPT" "IPOPT (B)" "ProxDDP"], [constrs_ipd constrs_ipo constrs_ipob constrs_al], title="Max Violation",
                yaxis=:log10, legend=false, size=(300, 300))
    savefig("plots/$problemclass/constr.pdf")

    boxplot(["IPDDP2" "IPOPT" "IPOPT (B)" "ProxDDP"], [iters_ipd iters_ipo iters_ipob iters_al], title="Iterations",
                yaxis=:log10, legend=false, size=(300, 300))
    savefig("plots/$problemclass/iterations.pdf")

    boxplot(["IPDDP2" "IPOPT" "IPOPT (B)"], [wall_ipd wall_ipo wall_ipob], legend=false, size=(300, 300),
                )
    savefig("plots/$problemclass/time.pdf")
end

plot_results("cartpole_friction")
plot_results("cartpole_friction_quad")
plot_results("concar")
plot_results("acrobot_contact")
plot_results("pushing_1_obs")
