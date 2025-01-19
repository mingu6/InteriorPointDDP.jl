using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using Printf

benchmark = false
verbose = true
quasi_newton = false
n_benchmark = 10

T = Float64
Δ = 0.01
N = 101
xN = T[1.0; 0.0]
x1 = T[0.0; 0.0]

options = Options{T}(quasi_newton=quasi_newton, verbose=true)
    
num_state = 2  # position and velocity
num_control = 3  # pushing force, 2x slacks for + and - components of abs work

# ## Dynamics - forward Euler

f = (x, u) -> x + Δ * [x[2], u[1]]

blockmove_dyn = Dynamics(f, num_state, num_control)
dynamics = [blockmove_dyn for k = 1:N-1]

# ## Objective

stage_cost = Objective((x, u) -> Δ * (u[2] + u[3]), 2, 3)
objective = [
    [stage_cost for k = 1:N-1]...,
    Objective((x, u) -> 500.0 * dot(x - xN, x - xN), 2, 0),
]

# ## Constraints

path_constr = Constraint((x, u) -> [
    u[2] - u[3] - u[1] * x[2]
], 2, 3)
constraints = [path_constr for k = 1:N-1]

# ## Bounds

bound = Bound(T[-10.0, 0.0, 0.0], T[10.0, Inf, Inf])
bounds = [bound for k = 1:N-1]

solver = Solver(T, dynamics, objective, constraints, bounds, options=options)

fname = quasi_newton ? "results/blockmove_QN.txt" : "results/blockmove.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for seed = 1:50
        solver.options.verbose = verbose
        Random.seed!(seed)
        
        # ## Initialise solver and solve
        
        ū = [[T(1.0e-0) * (randn(T, 1) .- 0.5); T(0.01) * ones(T, 2)] for k = 1:N-1]
        solve!(solver, x1, ū)

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
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", seed, solver.data.k, solver.data.status == 0,
                    solver.data.objective, solver.data.primal_inf, wall_time * 1000, solver_time * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        end
    end
end
