using IterativeLQR 
using LinearAlgebra
using Plots
using Random

N = 101
h = 0.05
r_car = 0.02
x0 = [0.0; 0.0; 0.0] 
xN = [1.0; 1.0; π / 2]

options = Options()
options.scaling_penalty = 4.0
options.initial_constraint_penalty = 1.0
options.objective_tolerance = 1e-5
options.lagrangian_gradient_tolerance = 1e-5
options.constraint_tolerance = 1e-3
options.max_iterations = 1000
options.max_dual_updates = 10

Random.seed!(0)

# ## car 
num_state = 3
num_action = 2

Random.seed!(0)

# ## control limits

ul = -1.0 * [1.0; 5.0]
uu = 1.0 * [1.0; 5.0]

# ## obstacles

xyr_obs = [
    [0.05, 0.25, 0.1],
    [0.45, 0.1, 0.15],
    [0.7, 0.7, 0.2],
    [0.35, 0.4, 0.1]
    ]
num_obstacles = length(r_obs)

# ## intermediate waypoints

xy_wp = [[0.2, 0.6], [0.7, 0.4]]
inds_wp = [40, 80]

include("../examples/visualise/concar.jl")

# ## Dynamics - explicit midpoint for integrator

function car_continuous(x, u)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function car_discrete(x, u)
    x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)
end

car = Dynamics(car_discrete, num_state, num_action)
dynamics = [car for k = 1:N-1] 

# ## objective 

stage_cost = (x, u) -> begin
    J = 0.0
    J += 1e-2 * dot(x - xN, x - xN)
    J += 1e-1 * dot(u, u)
    return J
end
objective = [
    [Cost(stage_cost, num_state, num_action) for k = 1:N-1]...,
    Cost((x, u) -> 1e3 * dot(x - xN, x - xN), num_state, 0)
]

# ## constraints - waypoints are constraints for iLQR

obs_dist(obs_xy) = (x, u) -> begin
    xy_diff = x[1:2] - obs_xy
    return dot(xy_diff, xy_diff)
end
stage_constr_fn = (x, u) -> begin
[
    ul - u; ## control limit (lower)
    u - uu; ## control limit (upper)
    # obstacle avoidance constraints i.e., d_thresh^2 - d_obs^2 <= 0 
    [(obs[3] + r_car)^2 - obs_dist(obs[1:2])(x, u)
        for (i, obs) in enumerate(xyr_obs)];
    # bound constraints, car must stay within [0, 1] x [0, 1] box
    -x[1];
    -x[2];
    x[1] - 1.0;
    x[2] - 1.0;
]
end

obs_constr = Constraint(stage_constr_fn, num_state, num_action,
indices_inequality=collect(1:2*num_action+n_obstacles+4))

# waypoint constraint

waypt_constr(p) = Constraint(
    (x, u) -> begin
    [
        ul - u; ## control limit (lower)
        u - uu; ## control limit (upper)
        p - x[1:2]
    ]
end, num_state, num_action,
indices_inequality=collect(1:2*num_action))

constraints = [[obs_constr for k = 1:N-1]..., Constraint()]
for (i, k) in enumerate(inds_wp)
    constraints[k] = waypt_constr(xy_wp[i])
end

# ## Initialise solver and solve

ū = [1.0e-1 * rand(2) for k = 1:N-1]
x̄ = rollout(dynamics, x0, ū)

solver = Solver(dynamics, objective, constraints; options=options)
initialize_controls!(solver, ū) 
initialize_states!(solver, x̄)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

# ## visualize
plot()
plotTrajectory!(x_sol)
for (xy, r) in zip(xy_obs, r_obs)
    plotCircle!(xy[1], xy[2], r)
end
scatter!(map(x -> x[1], waypoint_xy), map(x -> x[2], waypoint_xy), markershape=:star)
savefig("comparisons/plots/concar.png")

# ## benchmark allocations + timing
using BenchmarkTools
solver.options.verbose = false
info = @benchmark solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))