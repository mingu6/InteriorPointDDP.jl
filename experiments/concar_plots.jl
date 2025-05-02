using StatsPlots

problemclass = "concar"

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
ymax = 12.0

# objective function value

bplots = []
push!(bplots, boxplot(objs_ipd, title=names[1], titlefontsize=10, legend=false, ylims=(ymin, ymax), xticks=[]))
push!(bplots, boxplot(objs_ipo, title=names[2], titlefontsize=10, legend=false, ylims=(ymin, ymax), xticks=[]))
push!(bplots, boxplot(objs_ipob, title=names[3], titlefontsize=10, legend=false, ylims=(ymin, ymax),xticks=[]))
push!(bplots, boxplot(objs_al, title=names[4], titlefontsize=10, legend=false, ylims=(1e-2, 2000), yaxis=:log10,
            yticks=[1e-1, 1, 1e1, 1e2, 1e3], xticks=[]))
plot!(bplots..., size=(400, 300), layout=(1, 4))
savefig("plots/$problemclass/objective.pdf")

# constraint violation value

bplots = []
push!(bplots, boxplot(constrs_ipd, title=names[1], titlefontsize=10, legend=false, xticks=[], yaxis=:log10, ylims=(1e-16, 1e-12)))
push!(bplots, boxplot(constrs_ipo, title=names[2], titlefontsize=10, legend=false, xticks=[], yaxis=:log10, ylims=(1e-16, 1e-12)))
push!(bplots, boxplot(constrs_ipob, title=names[3], titlefontsize=10, legend=false, xticks=[], yaxis=:log10, ylims=(1e-16, 1e-12)))
push!(bplots, boxplot(constrs_al, title=names[4], titlefontsize=10, legend=false, yaxis=:log10,
            yticks=[1e-2, 1e-1, 1, 1e1], xticks=[], ylims=(0.5e-2, 2e1)))
plot(bplots..., size=(420, 300), layout=(1, 4))
savefig("plots/$problemclass/constr.pdf")

# iteration count

bplots = []
push!(bplots, boxplot(iters_ipd, title=names[1], titlefontsize=10, legend=false, xticks=[], ylims=(0, 550)))
push!(bplots, boxplot(iters_ipo, title=names[2], titlefontsize=10, legend=false, xticks=[], ylims=(0, 550)))
push!(bplots, boxplot(iters_ipob, title=names[3], titlefontsize=10, legend=false, xticks=[], ylims=(0, 550)))
push!(bplots, boxplot(iters_al, title=names[4], titlefontsize=10, legend=false, xticks=[], ylims=(0, 2200)))
plot(bplots..., size=(400, 300), layout=(1, 4))
savefig("plots/$problemclass/iterations.pdf")

# wall time

bplots = []
push!(bplots, boxplot(wall_ipd, title=names[1], titlefontsize=10, legend=false, xticks=[], ylims=(0, 550)))
push!(bplots, boxplot(wall_ipo, title=names[2], titlefontsize=10, legend=false, xticks=[], ylims=(0, 550)))
push!(bplots, boxplot(wall_ipob, title=names[3], titlefontsize=10, legend=false, xticks=[], ylims=(0, 1550)))
plot(bplots..., size=(300, 300), layout=(1, 3))
savefig("plots/$problemclass/time.pdf")