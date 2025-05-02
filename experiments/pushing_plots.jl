using StatsPlots

problemclass = "pushing_1_obs"

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
ymax = 0.3

# objective function value

bplots = []
push!(bplots, boxplot(objs_ipd, title=names[1], titlefontsize=10, legend=false, yaxis=:log10, ylims=(1e-2, 1e1), xticks=[]))
push!(bplots, boxplot(objs_ipo, title=names[2], titlefontsize=10, legend=false, yaxis=:log10, ylims=(1e-2, 1e1), xticks=[]))
push!(bplots, boxplot(objs_ipob, title=names[3], titlefontsize=10, legend=false, yaxis=:log10, ylims=(1e-2, 1e1), xticks=[]))
push!(bplots, boxplot(objs_al, title=names[4], titlefontsize=10, legend=false, xticks=[], ylims=(1e-2, 1e1), yaxis=:log10))
plot!(bplots..., size=(400, 300), layout=(1, 4))
savefig("plots/$problemclass/objective.pdf")

# constraint violation value

bplots = []
push!(bplots, boxplot(constrs_ipd, title=names[1], titlefontsize=10, legend=false, xticks=[], yaxis=:log10,
                        ylims=(1e-18, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
push!(bplots, boxplot(constrs_ipo, title=names[2], titlefontsize=10, legend=false, xticks=[], yaxis=:log10,
                        ylims=(1e-18, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
push!(bplots, boxplot(constrs_ipob, title=names[3], titlefontsize=10, legend=false, xticks=[], yaxis=:log10,
                        ylims=(1e-18, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
push!(bplots, boxplot(constrs_al, title=names[4], titlefontsize=10, legend=false, yaxis=:log10, xticks=[],
                        ylims=(1e-18, 1e1), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1]))
plot(bplots..., size=(420, 300), layout=(1, 4))
savefig("plots/$problemclass/constr.pdf")

# iteration count

bplots = []
push!(bplots, boxplot(iters_ipd, title=names[1], titlefontsize=10, legend=false, xticks=[], ylims=(0, 1050)))
push!(bplots, boxplot(iters_ipo, title=names[2], titlefontsize=10, legend=false, xticks=[], ylims=(0, 1050)))
push!(bplots, boxplot(iters_ipob, title=names[3], titlefontsize=10, legend=false, xticks=[], ylims=(0, 1050)))
push!(bplots, boxplot(iters_al, title=names[4], titlefontsize=10, legend=false, xticks=[], ylims=(0, 1050)))
plot(bplots..., size=(400, 300), layout=(1, 4))
savefig("plots/$problemclass/iterations.pdf")

# wall time

bplots = []
push!(bplots, boxplot(wall_ipd, title=names[1], titlefontsize=10, legend=false, xticks=[],
                        ylims=(0, 3000), yticks=[0, 500, 1000, 1500, 2000, 2500, 3000]))
push!(bplots, boxplot(wall_ipo, title=names[2], titlefontsize=10, legend=false, xticks=[],
                        ylims=(0, 3000), yticks=[0, 500, 1000, 1500, 2000, 2500, 3000]))
push!(bplots, boxplot(wall_ipob, title=names[3], titlefontsize=10, legend=false, xticks=[],
                        ylims=(0, 3000), yticks=[0, 500, 1000, 1500, 2000, 2500, 3000]))
plot(bplots..., size=(320, 300), layout=(1, 3))
savefig("plots/$problemclass/time.pdf")