using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using BenchmarkTools
using Printf

visualise = false
benchmark = true
verbose = true

T = Float64
h = 0.01
N = 101
xN = T[1.0; 0.0]
x1 = T[0.0; 0.0]

options = Options{T}(quasi_newton=false, verbose=true)
    
num_state = 2  # position and velocity
num_control = 3  # pushing force, 2x slacks for + and - components of abs work

# ## Dynamics - explicit midpoint for integrator

function blockmove_continuous(x, u)
    return [x[2], u[1]]
end
blockmove_discrete = (x, u) -> x + h * blockmove_continuous(x + 0.5 * h * blockmove_continuous(x, u), u)

blockmove_dyn = Dynamics(blockmove_discrete, num_state, num_control)
dynamics = [blockmove_dyn for k = 1:N-1]

# ## Costs

stage_cost = Cost((x, u) -> h * (u[2] + u[3]), 2, 3)
objective = [
    [stage_cost for k = 1:N-1]...,
    Cost((x, u) -> 500.0 * dot(x - xN, x - xN), 2, 0),
]

# ## Constraints

stage_constr = Constraint((x, u) -> [
    u[2] - u[3] - u[1] * x[2]
], 2, 3)
constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(T[-10.0, 0.0, 0.0], T[10.0, Inf, Inf])
bounds = [bound for k = 1:N-1]

solver = Solver(T, dynamics, objective, constraints, bounds, options=options)

open("examples/results/blockmove.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        time (s)  \n")
    for seed = 1:50
        solver.options.verbose = verbose
        Random.seed!(seed)
        
        # ## Initialise solver and solve
        
        ū = [[T(1.0e-0) * (randn(T, 1) .- 0.5); T(0.01) * ones(T, 2)] for k = 1:N-1]
        solve!(solver, x1, ū)

        if benchmark
            solver.options.verbose = false
            solve_time = @belapsed solve!($solver, $x1, $ū)
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %.5f  \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf, solve_time)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        end
    end
end

# ## Plot solution

if visualise
    x_sol, u_sol = get_trajectory(solver)
    
    x = map(x -> x[1], x_sol)
    v = map(x -> x[2], x_sol)
    u = [map(u -> u[1], u_sol); 0.0]
    work = [abs(vk * uk) for (vk, uk) in zip(v, u)]
    plot(range(0, (N-1) * h, length=N), [x v u work], label=["x" "v" "u" "work"])
    savefig("examples/plots/blockmove.png")
    
    println("Total absolute work: ", sum(work))
end
