using StatsPlots
using Statistics

problemclass = "acrobot_contact"
fs = 16
fs_y = 12

include("utils.jl")

names = ["IPDDP2" "IPOPT" "IPOPT (B)" "ProxDDP"]

_, iters_ipd, _, objs_ipd, constrs_ipd, wall_ipd, _ = read_results("ipddp2/results/$problemclass.txt")
_, iters_ipo, _, objs_ipo, constrs_ipo, wall_ipo, _ = read_results("ipopt/results/$problemclass.txt")
_, iters_ipob, _, objs_ipob, constrs_ipob, wall_ipob, _ = read_results("ipopt/results/bfgs_$problemclass.txt")
_, iters_al, _, objs_al, constrs_al, _, _ = read_results("proxddp/results/$problemclass.txt")

min_objs = min(minimum(objs_ipd), minimum(objs_ipo))
max_objs = max(maximum(objs_ipd), maximum(objs_ipo))
objs_range = max_objs - min_objs
ymin = 0.0
ymax = 2.8

# objective function value

bplots = []
push!(bplots, boxplot(objs_ipd, title=names[1], titlefontsize=fs, ytickfontsize=fs_y, legend=false, yaxis=:log10, ylims=(1e-1, 3e4), xticks=[]))
push!(bplots, boxplot(objs_ipo, title=names[2], titlefontsize=fs, ytickfontsize=fs_y, legend=false, yaxis=:log10, ylims=(1e-1, 3e4), xticks=[]))
push!(bplots, boxplot(objs_ipob, title=names[3], titlefontsize=fs, ytickfontsize=fs_y, legend=false, yaxis=:log10, ylims=(1e-1, 3e4), xticks=[]))
push!(bplots, boxplot(objs_al, title=names[4], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], yaxis=:log10, ylims=(1e-1, 3e4)))
plot!(bplots..., size=(650, 350), layout=(1, 4))
savefig("plots/$problemclass/objective0.svg")

# constraint violation value

bplots = []
push!(bplots, boxplot(constrs_ipd, title=names[1], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], yaxis=:log10,
                        ylims=(1e-16, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
push!(bplots, boxplot(constrs_ipo, title=names[2], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], yaxis=:log10,
                        ylims=(1e-16, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
push!(bplots, boxplot(constrs_ipob, title=names[3], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], yaxis=:log10,
                        ylims=(1e-16, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
push!(bplots, boxplot(constrs_al, title=names[4], titlefontsize=fs, ytickfontsize=fs_y, legend=false, yaxis=:log10, xticks=[],
                        ylims=(1e-16, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
plot(bplots..., size=(650, 350), layout=(1, 4))
savefig("plots/$problemclass/constr0.svg")

# iteration count

bplots = []
push!(bplots, boxplot(iters_ipd, title=names[1], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], ylims=(0, 1550)))
push!(bplots, boxplot(iters_ipo, title=names[2], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], ylims=(0, 1550)))
push!(bplots, boxplot(iters_ipob, title=names[3], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], ylims=(0, 1550)))
push!(bplots, boxplot(iters_al, title=names[4], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], ylims=(0, 1550)))
plot(bplots..., size=(650, 350), layout=(1, 4))
savefig("plots/$problemclass/iterations0.svg")

# wall time per iteration

bplots = []
push!(bplots, boxplot(wall_ipd ./ iters_ipd, title=names[1], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], ylims=(0, 4.9)))
push!(bplots, boxplot(wall_ipo ./ iters_ipo, title=names[2], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], ylims=(0, 4.9)))
push!(bplots, boxplot(wall_ipob ./ iters_ipob, title=names[3], titlefontsize=fs, ytickfontsize=fs_y, legend=false, xticks=[], ylims=(0, 4.9)))
plot(bplots..., size=(500, 350), layout=(1, 3))
savefig("plots/$problemclass/time0.svg")

println("Acrobot")
println()
println("Objective (rel IPOPT): ", median(objs_ipd ./ objs_ipo))
println("Iteration Count (rel IPOPT): ", median(iters_ipd ./ iters_ipo))
println("Wall Time per iteration (rel IPOPT): ", median((wall_ipd ./ iters_ipd) ./ (wall_ipo ./ iters_ipo)))
println()
println("Objective (rel IPOPT BFGS): ", median(objs_ipd ./ objs_ipob))
println("Iteration Count (rel IPOPT BFGS): ", median(iters_ipd ./ iters_ipob))
println("Wall Time per iteration (rel IPOPT BFGS): ", median((wall_ipd ./ iters_ipd) ./ (wall_ipob ./ iters_ipob)))
println()
println("Objective (rel ProxDDP): ", median(objs_ipd ./ objs_al))
println("Iteration Count (rel ProxDDP): ", median(iters_ipd ./ iters_al))
println()
