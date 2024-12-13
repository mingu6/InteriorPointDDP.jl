using InteriorPointDDP
using LinearAlgebra
using Plots
using Random
using Printf

visualise = true
benchmark = false
verbose = true
quasi_newton = false
n_benchmark = 10

T = Float64
N = 101
h = 0.05
r_car = 0.02
xN = T[1.0; 1.0; π / 4; 0.0]
options = Options{T}(quasi_newton=quasi_newton, verbose=true)

num_state = 4
num_action = 2

# ## control limits

ul = T[-2.0; -5.0]
uu = T[2.0; 5.0]

# ## obstacles

xyr_obs = [
    T[0.05, 0.25, 0.1],
    T[0.45, 0.1, 0.15],
    T[0.7, 0.7, 0.2],
    T[0.30, 0.4, 0.1]
    ]
num_obstacles = length(xyr_obs)
num_primal = num_action + num_obstacles + 2  # 2 slacks for box constraints

include("../examples/visualise/concar.jl")

# ## Dynamics - explicit midpoint for integrator

function car_continuous(x, u)
    [x[4] * cos(x[3]); x[4] * sin(x[3]); u[2]; u[1]]
end

function car_discrete(x, u)
    x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)
end

car = Dynamics(car_discrete, num_state, num_primal)
dynamics = [car for k = 1:N-1]

# ## objective - waypoint constraints -> high cost

stage_cost = (x, u) -> begin
    J = 0.0
    J += h * dot(x - xN, x - xN)
    J += h * dot(u[1:2] .* [10.0, 1.0], u[1:2])
    return J
end
objective = [
    [Cost(stage_cost, num_state, num_primal) for k = 1:N-1]...,
    Cost((x, u) -> 1e3 * dot(x - xN, x - xN), num_state, 0)
]

# ## constraints

obs_dist(obs_xy) = (x, u) -> begin
    xp = car_discrete(x, u)[1:2]
    xy_diff = xp[1:2]- obs_xy
    return dot(xy_diff, xy_diff)
end
stage_constr_fn = (x, u) -> begin
[
    # obstacle avoidance constraints w/slack variable,
    # i.e., d_obs^2 - d_thresh^2 >= 0 and d_obs^2 - d_thresh^2 + s = 0, s >= 0
    [(obs[3] + r_car)^2 - obs_dist(obs[1:2])(x, u) + u[num_action + i]
        for (i, obs) in enumerate(xyr_obs)];
    # bound constraints, car must stay within [0, 1] x [0, 1] box
    car_discrete(x, u)[1:2] - u[end-1:end]
]
end

obs_constr = Constraint(stage_constr_fn, num_state, num_primal)
constraints = [obs_constr for k = 1:N-1]

# ## bounds

# [control limits; obs slack; bound slack]
bound = Bound(
    [ul; zeros(T, num_obstacles); zeros(T, 2)],
    [uu; T(Inf) * ones(T, num_obstacles); ones(T, 2)]
)
bounds = [bound for k in 1:N-1]

solver = Solver(T, dynamics, objective, constraints, bounds, options=options)

# ## Plots

if visualise
    plot(xlims=(0, 1), ylims=(0, 1), xtickfontsize=14, ytickfontsize=14)
    for xyr in xyr_obs
        plotCircle!(xyr[1], xyr[2], xyr[3])
    end
end

# ## Initialise solver and solve

fname = quasi_newton ? "examples/results/concar_QN.txt" : "examples/results/concar.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for seed = 1:50
        solver.options.verbose = verbose
        Random.seed!(seed)
        
        x1 = T[0.0; 0.0; 0.0; 0.0] + rand(T, 4) .* T[0.05; 0.05; π / 2; 0.0]
        ū = [[T(1.0e-3) .* (rand(T, 2) .- 0.5); T(0.01) * ones(T, num_obstacles + 2)] for k = 1:N-1]
    
        solve!(solver, x1, ū)
        
        if visualise
            x_sol, u_sol = get_trajectory(solver)
            plotTrajectory!(x_sol)
        end
        
        if benchmark
            solver.options.verbose = false
            solver_time = 0.0
            wall_time = 0.0
            for i in 1:n_benchmark
                solve!(solver, x1, ū)
                solver_time += solver.data.solver_time
                wall_time += solver.data.wall_time
            end
            solver_time /= n_benchmark
            wall_time /= n_benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f         %5.1f  \n", seed, solver.data.k, solver.data.status == 0,
                    solver.data.objective, solver.data.primal_inf, wall_time * 1000, solver_time * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        end
    end
end

visualise && savefig("examples/plots/concar_IPDDP.pdf")
