using Printf
using Statistics

function read_results(fname::String)
    regex_results = r"(\d+)(\s+)(\d+)(\s+)(\w+)(\s+)(\d+.\d+e[+-]\d+)(\s+)(\d+.\d+e[+-]\d+)(\s+)(\d+.\d+)  "
    
    seeds = Int64[]
    iters = Int64[]
    statuss = Bool[]
    objs = Float64[]
    constrs = Float64[]
    timings = Float64[]
    
    open(fname, "r") do io
        lines = readlines(io)
        
        for line in lines
            res = match(regex_results, line)
            if !isnothing(res)
                seed, iter, status, obj, constr, timing = parse_results(res.captures)
                push!(seeds, seed)
                push!(iters, iter)
                push!(statuss, status)
                push!(objs, obj)
                push!(constrs, constr)
                push!(timings, timing)
            end
        end
    end
    
    return seeds, iters, statuss, objs, constrs, timings
end

function parse_results(res)
    seed = parse(Int64, res[1])
    iters = parse(Int64, res[3])
    status = parse(Bool, res[5])
    obj = parse(Float64, res[7])
    constr = parse(Float64, res[9])
    timing = parse(Float64, res[11])
    return seed, iters, status, obj, constr, timing
end

fnames_in = ["../examples/results/concar.txt", "ipopt/results/concar.txt", "ALiLQR/results/concar.txt"]
names = ["IPDDP2", "IPOPT", "AL-iLQR"]

open("summary_results_concar.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        @printf(io, "%s \n", name)
        @printf(io, "\n")
        seeds, iters, statuss, objs, constrs, timings = read_results(fname)
        @printf(io, "Iterations:  %s  %s  %s \n", quantile(iters, 0.25), median(iters), quantile(iters, 0.75))
        @printf(io, "Objective:  %.8e  %.8e  %.8e \n", quantile(objs, 0.25), median(objs), quantile(objs, 0.75))
        @printf(io, "Constraint Violation:  %.8e  %.8e  %.8e \n", quantile(constrs, 0.25), median(constrs), quantile(constrs, 0.75))
        @printf(io, "Wall Clock (s):  %.5f  %.5f  %.5f \n", quantile(timings, 0.25), median(timings), quantile(timings, 0.75))
        @printf(io, "\n")
    end
end

fnames_in = ["../examples/results/cartpole_implicit.txt", "ipopt/results/cartpole.txt", "ALiLQR/results/cartpole_implicit.txt"]
names = ["IPDDP2", "IPOPT", "AL-iLQR"]

open("summary_results_cartpole_implicit.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        @printf(io, "%s \n", name)
        @printf(io, "\n")
        seeds, iters, statuss, objs, constrs, timings = read_results(fname)
        @printf(io, "Iterations:  %s  %s  %s \n", quantile(iters, 0.25), median(iters), quantile(iters, 0.75))
        @printf(io, "Objective:  %.8e  %.8e  %.8e \n", quantile(objs, 0.25), median(objs), quantile(objs, 0.75))
        @printf(io, "Constraint Violation:  %.8e  %.8e  %.8e \n", quantile(constrs, 0.25), median(constrs), quantile(constrs, 0.75))
        @printf(io, "Wall Clock (s):  %.5f  %.5f  %.5f \n", quantile(timings, 0.25), median(timings), quantile(timings, 0.75))
        @printf(io, "\n")
    end
end

fnames_in = ["../examples/results/blockmove.txt", "ipopt/results/blockmove.txt", "ALiLQR/results/blockmove.txt"]
names = ["IPDDP2", "IPOPT", "AL-iLQR"]

open("summary_results_blockmove.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        @printf(io, "%s \n", name)
        @printf(io, "\n")
        seeds, iters, statuss, objs, constrs, timings = read_results(fname)
        @printf(io, "Iterations:  %s  %s  %s \n", quantile(iters, 0.25), median(iters), quantile(iters, 0.75))
        @printf(io, "Objective:  %.8e  %.8e  %.8e \n", quantile(objs, 0.25), median(objs), quantile(objs, 0.75))
        @printf(io, "Constraint Violation:  %.8e  %.8e  %.8e \n", quantile(constrs, 0.25), median(constrs), quantile(constrs, 0.75))
        @printf(io, "Wall Clock (s):  %.5f  %.5f  %.5f \n", quantile(timings, 0.25), median(timings), quantile(timings, 0.75))
        @printf(io, "\n")
    end
end

fnames_in = ["../examples/results/acrobot_contact.txt", "ipopt/results/acrobot_contact.txt", "ALiLQR/results/acrobot_contact.txt"]
names = ["IPDDP2", "IPOPT", "AL-iLQR"]

open("summary_results_acrobot_contact.txt", "w") do io
    for (fname, name) in zip(fnames_in, names)
        @printf(io, "%s \n", name)
        @printf(io, "\n")
        seeds, iters, statuss, objs, constrs, timings = read_results(fname)
        @printf(io, "Iterations:  %s  %s  %s \n", quantile(iters, 0.25), median(iters), quantile(iters, 0.75))
        @printf(io, "Objective:  %.8e  %.8e  %.8e \n", quantile(objs, 0.25), median(objs), quantile(objs, 0.75))
        @printf(io, "Constraint Violation:  %.8e  %.8e  %.8e \n", quantile(constrs, 0.25), median(constrs), quantile(constrs, 0.75))
        @printf(io, "Wall Clock (s):  %.5f  %.5f  %.5f \n", quantile(timings, 0.25), median(timings), quantile(timings, 0.75))
        @printf(io, "\n")
    end
end


# function plot_iteration_count(fnames_in::Vector{String}, names::Vector{String})
#     plot()
#     for (fname, name) in zip(fnames_in, names)
#         seeds, iters, statuss, objs, constrs, timings = read_results(fname)
#         histogram!(iters, linecolor=1, xtickfontsize=14, ytickfontsize=14, legendfontsize=14, bins=range(150, 400, length=10), label=name)
#     end
#     savefig("iteration_count.png")
# end

# function plot_objective(fnames_in::Vector{String}, names::Vector{String})
#     plot()
#     for (fname, name) in zip(fnames_in, names)
#         seeds, iters, statuss, objs, constrs, timings = read_results(fname)
#         histogram!(objs, linecolor=1, xtickfontsize=14, ytickfontsize=14, legendfontsize=14, bins=15, label=name)
#     end
#     savefig("objective.png")
# end

# function plot_violation(fnames_in::Vector{String}, names::Vector{String})
#     plot()
#     for (fname, name) in zip(fnames_in, names)
#         seeds, iters, statuss, objs, constrs, timings = read_results(fname)
#         clamp!(constrs, 1e-16, Inf)
#         histogram!(constrs, linecolor=1, xtickfontsize=14, ytickfontsize=14, legendfontsize=14, bins=15, label=name)
#     end
#     savefig("violation.png")
# end

# # plot_iteration_count(["../examples/results/concar.txt", "ipopt/results/concar.txt", "ALiLQR/results/concar.txt"])
# plot_iteration_count(["../examples/results/concar.txt", "ALiLQR/results/concar.txt"], ["IPDDP2", "AL-iLQR"])
# plot_objective(["../examples/results/concar.txt", "ALiLQR/results/concar.txt"], ["IPDDP2", "AL-iLQR"])
# plot_violation(["../examples/results/concar.txt", "ALiLQR/results/concar.txt"], ["IPDDP2", "AL-iLQR"])

# write_results(["../examples/results/concar.txt", "ALiLQR/results/concar.txt"], ["IPDDP2", "AL-iLQR"])