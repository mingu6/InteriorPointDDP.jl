using Printf
using Statistics

function read_results(fname::String)
    regex_results = r"\s*(\d+)\s+(\d+)\s+(\w+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+)\s+(\d+.\d+)"
    
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
    wall = parse(Float64, res[6])
    solver = parse(Float64, res[7])
    return seed, iters, status, obj, constr, wall, solver
end

function print_summary_results(io, fname, alg_name)
    @printf(io, "%s \n", alg_name)
    @printf(io, "\n")
    _, iters, statuss, objs, constrs, wall, solver = read_results(fname)
    @printf(io, "Iterations:  %s  %s  %s  %s  %s  \n", minimum(iters), quantile(iters, 0.25), median(iters), quantile(iters, 0.75), maximum(iters))
    @printf(io, "Objective:  %.8e  %.8e  %.8e  %.8e  %.8e  \n", minimum(objs), quantile(objs, 0.25), median(objs), quantile(objs, 0.75), maximum(objs))
    @printf(io, "Constraint Violation:  %.8e  %.8e  %.8e  %.8e  %.8e  \n", minimum(constrs), quantile(constrs, 0.25), median(constrs), quantile(constrs, 0.75), maximum(constrs))
    @printf(io, "Wall (ms):  %5.1f  %5.1f  %5.1f  %5.1f  %5.1f \n", minimum(wall), quantile(wall, 0.25), median(wall), quantile(wall, 0.75), maximum(wall))
    @printf(io, "Solver (ms):  %5.1f  %5.1f  %5.1f  %5.1f  %5.1f  \n", minimum(solver) ,quantile(solver, 0.25), median(solver), quantile(solver, 0.75), maximum(solver))
    @printf(io, "Num. Failed:  %s, Num. Success:  %s", length(statuss) - sum(statuss), sum(statuss))
    @printf(io, "\n")
    @printf(io, "\n")
end

fnames_in = ["ipddp2results/concar.txt", "ipddp2/results/concar_QN.txt", "ipopt/results/concar.txt", "ALiLQR/results/concar.txt"]
names = ["IPDDP2", "IPDDP2 (QN)", "IPOPT", "AL-iLQR"]

open("summary_results_concar.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        print_summary_results(io, fname, name)
    end
end

fnames_in = ["ipddp2/results/cartpole_implicit.txt", "ipddp2/results/cartpole_implicit_QN.txt", "ipopt/results/cartpole.txt", "ALiLQR/results/cartpole_implicit.txt"]

open("summary_results_cartpole_implicit.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        print_summary_results(io, fname, name)
    end
end

fnames_in = ["ipddp2/results/blockmove.txt", "ipddp2/results/blockmove_QN.txt", "ipopt/results/blockmove.txt", "ALiLQR/results/blockmove.txt"]

open("summary_results_blockmove.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        print_summary_results(io, fname, name)
    end
end

fnames_in = ["ipddp2/results/acrobot_contact.txt", "ipddp2/results/acrobot_contact_QN.txt", "ipopt/results/acrobot_contact.txt", "ALiLQR/results/acrobot_contact.txt"]

open("summary_results_acrobot_contact.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        print_summary_results(io, fname, name)
    end
end
