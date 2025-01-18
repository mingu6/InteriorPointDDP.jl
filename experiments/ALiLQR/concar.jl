using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using BenchmarkTools
using Printf

benchmark = true
verbose = true

N = 101
Δ = 0.05
r_car = 0.02
xN = [1.0; 1.0; π / 4; 0.0]

options = Options()
options.scaling_penalty = 10.0
options.initial_constraint_penalty = 1.0
options.max_dual_updates = 300
options.constraint_tolerance = 1e-8
options.constraint_norm = 1

# ## car 
num_state = 4
num_action = 2

# ## control limits

ul = [-2.0; -4.0]
uu = [2.0; 4.0]

# ## obstacles

xyr_obs = [
    [0.05, 0.25, 0.1],
    [0.45, 0.1, 0.15],
    [0.7, 0.7, 0.2],
    [0.30, 0.4, 0.1]
    ]
num_obstacles = length(xyr_obs)

include("../visualise/concar.jl")

# ## Dynamics - RK4

# continuous time dynamics
function g(x, u)
    [x[4] * cos(x[3]); x[4] * sin(x[3]); u[2]; u[1]]
end

function RK4(x, u, g)
    k1 = g(x, u)
    k2 = g(x + Δ * 0.5 * k1, u)
    k3 = g(x + Δ * 0.5 * k2, u)
    k4 = g(x + Δ * k3, u)
    return x + Δ / 6 * (k1 + k2 + k3 + k4)
end

f = (x, u) -> RK4(x, u, g)

car = Dynamics(f, num_state, num_action)
dynamics = [car for k = 1:N-1] 

# ## objective 

stage_cost = (x, u) -> begin
    J = 0.0
    J += Δ * dot(x - xN, x - xN)
    J += Δ * dot(u[1:2] .* [10.0, 1.0], u[1:2])
    return J
end
objective = [
    [Cost(stage_cost, num_state, num_action) for k = 1:N-1]...,
    Cost((x, u) -> 1e3 * dot(x - xN, x - xN), num_state, 0)
]

# ## constraints - waypoints are constraints for iLQR

obs_dist(obs_xy) = (x) -> begin
    xy_diff = x[1:2] - obs_xy
    return dot(xy_diff, xy_diff)
end
stage_constr_fn = (x, u) -> begin
[
    ul - u; ## control limit (lower)
    u - uu; ## control limit (upper)
    # obstacle avoidance constraints i.e., d_thresh^2 - d_obs^2 <= 0 
    [(obs[3] + r_car)^2 - obs_dist(obs[1:2])(f(x, u))
        for (i, obs) in enumerate(xyr_obs)];
    # bound constraints, car must stay within [0, 1] x [0, 1] box
    -f(x, u)[1];
    -f(x, u)[2];
    f(x, u)[1] - 1.0;
    f(x, u)[2] - 1.0;
]

end

obs_constr = Constraint(stage_constr_fn, num_state, num_action,
    indices_inequality=collect(1:2*num_action+num_obstacles+4))

constraints = [[obs_constr for k = 1:N-1]..., Constraint()]

# ## Initialise solver

solver = Solver(dynamics, objective, constraints; options=options)

# ## Plots

plot(xlims=(0, 1), ylims=(0, 1), xtickfontsize=14, ytickfontsize=14)
for xyr in xyr_obs
    plotCircle!(xyr[1], xyr[2], xyr[3])
end

open("results/concar.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms)  \n")
	for seed = 1:50
        solver = Solver(dynamics, objective, constraints; options=options)
		solver.options.verbose = verbose
		Random.seed!(seed)
		
        # ## Initialise solver and solve
        
        x0 = rand(4) .* [0.05; 0.05; π / 2; 0.0]
        ū = [1.0e-3 .* (rand(2) .- 0.5) for k = 1:N-1]
        x̄ = rollout(dynamics, x0, ū)
        
        solve!(solver, x̄, ū)
        
        x_sol, u_sol = get_trajectory(solver)
        plotTrajectory!(x_sol)
		
		if benchmark
            solver.options.verbose = false
            solve_time = @belapsed solve!($solver, $x̄, $ū) samples=10 setup=(solver=Solver(dynamics, objective, constraints; options=options))
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %5.1f       %5.1f\n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1],
                            solver.data.max_violation[1], solve_time * 1000, 0.0)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1], solver.data.max_violation[1])
        end
    end
end

savefig("plots/concar_ALiLQR.pdf")
