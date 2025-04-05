using Printf
using Statistics

function read_results(fname::String)
    regex_results = r"\s*(\d+)\s+(\d+)\s+(\w+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+)\s+(\d+.\d+)"
    regex_no_bm = r"\s*(\d+)\s+(\d+)\s+(\w+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+e[+-]\d+)"
    
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

# function plot_results(problemclass)
#     fnames_in = ["ipddp2/results/$problemclass.txt", "ipopt/results/$problemclass.txt", "ALiLQR/results/$problemclass.txt"]
#     names = ["IPDDP2", "IPOPT", "AL-iLQR"]
#     metric = []
#     method = []
#     vals = []
#     for (fname, name) in zip(fnames_in, names)
#         _, iters, statuss, objs, constrs, wall, solver = read_results(fname)
#         vals = [vals; objs; constrs; iters; wall]
#         metric = [metric; ["Obj." for _ in 1:length(objs)]; ["Constr." for _ in 1:length(constrs)]; ["Iters." for _ in 1:length(iters)]; ["Time" for _ in 1:length(wall)]]
#         method = [method; [name for _ in 1:length([objs; constrs; iters; wall])]]
#     end
#     groupedboxplot(metric, vals, group=method, legend=:bottom, bar_width=0.7)
#     savefig("$problemclass.pdf")
# end

# function plot_results(problemclass)
#     fnames_in = ["ipddp2/results/$problemclass.txt", "ipopt/results/$problemclass.txt", "ALiLQR/results/$problemclass.txt"]
#     names = ["IPDDP2", "IPOPT", "AL-iLQR"]

#     _, iters_ipd, _, objs_ipd, constrs_ipd, wall_ipd, _ = read_results("ipddp2/results/$problemclass.txt")
#     _, iters_ipo, _, objs_ipo, constrs_ipo, wall_ipo, _ = read_results("ipopt/results/$problemclass.txt")
#     _, iters_al, _, objs_al, constrs_al, wall_al, _ = read_results("ALiLQR/results/$problemclass.txt")

#     boxplot(["IPDDP2" "IPOPT" "AL-iLQR"], [objs_ipd objs_ipo objs_al], title="Objective Function",
#         yaxis=:log10, legend=false, size=(240, 400))
#     savefig("objective_$problemclass.pdf")

#     boxplot(["IPDDP2" "IPOPT" "AL-iLQR"], [constrs_ipd constrs_ipo constrs_al], title="Max Violation", yaxis=:log10, legend=false, size=(240, 400))
#     savefig("constr_$problemclass.pdf")

#     boxplot(["IPDDP2" "IPOPT" "AL-iLQR"], [iters_ipd iters_ipo iters_al], title="Iterations", yaxis=:log10, legend=false, size=(240, 400))
#     savefig("iterations_$problemclass.pdf")

#     boxplot(["IPDDP2" "IPOPT" "AL-iLQR"], [wall_ipd wall_ipo wall_al], title="Total time (ms)", yaxis=:log10, legend=false, size=(240, 400))
#     savefig("time_$problemclass.pdf")
# end

function plot_results(problemclass)
    fnames_in = ["ipddp2/results/$problemclass.txt", "ipopt/results/$problemclass.txt"]
    names = ["IPDDP2", "IPOPT", "AL-iLQR"]

    _, iters_ipd, _, objs_ipd, constrs_ipd, wall_ipd, _ = read_results("ipddp2/results/$problemclass.txt")
    _, iters_ipo, _, objs_ipo, constrs_ipo, wall_ipo, _ = read_results("ipopt/results/$problemclass.txt")

    boxplot(["IPDDP2" "IPOPT"], [objs_ipd objs_ipo], title="Objective Function",
        yaxis=:log10, legend=false, size=(200, 300))
    savefig("objective_$problemclass.pdf")

    boxplot(["IPDDP2" "IPOPT"], [constrs_ipd constrs_ipo], title="Max Violation", yaxis=:log10, legend=false, size=(200, 300))
    savefig("constr_$problemclass.pdf")

    boxplot(["IPDDP2" "IPOPT"], [iters_ipd iters_ipo], title="Iterations", legend=false, size=(200, 300))
    savefig("iterations_$problemclass.pdf")

    boxplot(["IPDDP2" "IPOPT"], [wall_ipd wall_ipo], title="Total time (ms)", legend=false, size=(200, 300))
    savefig("time_$problemclass.pdf")
end

# plot_results("cartpole_inverse")
# plot_results("concar")
plot_results("pushing")
