using InteriorPointDDP
using LinearAlgebra
using Plots
using Random

Random.seed!(0)
T = Float64
N = 101
h = 0.05
r_car = 0.02
x1 = T[0.0; 0.0; 0.0] + rand(T, 3) .* T[0.05, 0.05, π / 2]
xN = T[1.0; 1.0; π / 4]
options = Options{T}(quasi_newton=false, verbose=true)

num_state = 3
num_action = 2

# ## control limits

ul = T[-0.1; -5.0]
uu = T[1.0; 5.0]

# ## obstacles

xyr_obs = [
    T[0.05, 0.25, 0.1],
    T[0.45, 0.1, 0.15],
    T[0.7, 0.7, 0.2],
    T[0.35, 0.4, 0.1]
    ]
num_obstacles = length(xyr_obs)
num_primal = num_action + num_obstacles + 2  # 2 slacks for box constraints

include("../examples/visualise/concar.jl")

# ## Dynamics - explicit midpoint for integrator

function car_continuous(x, u)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function car_discrete(x, u)
    x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)
end

car = Dynamics(car_discrete, num_state, num_primal)
dynamics = [car for k = 1:N-1]

# ## objective - waypoint constraints -> high cost

stage_cost = (x, u) -> begin
    J = 0.0
    J += 1e-2 * dot(x - xN, x - xN)
    J += 1e-1 * dot(u[1:2] .* [1.0, 0.1], u[1:2])
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

# ## Initialise solver and solve

ū = [[T(1.0e-2) .* (rand(T, 2) .- 0.5); T(0.1) * ones(T, num_obstacles + 2)] for k = 1:N-1]
solver = Solver(T, dynamics, objective, constraints, bounds, options=options)
solve!(solver, x1, ū)

# # ## Plot solution

x_sol, u_sol = get_trajectory(solver)

plot()
plotTrajectory!(x_sol)
for xyr in xyr_obs
    plotCircle!(xyr[1], xyr[2], xyr[3])
end
savefig("examples/plots/concar.png")

# ## benchmark allocations + timing
using BenchmarkTools
solver.options.verbose = false
@benchmark solve!($solver, $x1, $ū)
