using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using Printf

visualise = false
benchmark = true
verbose = true
quasi_newton = false
n_benchmark = 10

T = Float64
h = 0.01
N = 101
xN = T[1.0; 0.0]
x1 = T[0.0; 0.0]

options = Options{T}(quasi_newton=quasi_newton, verbose=true)
    
num_state = 2  # position and velocity
num_control = 3  # pushing force, 2x slacks for + and - components of abs work

# ## Dynamics - explicit midpoint for integrator

function blockmove_continuous(x, u)
    return [x[2], u[1]]
end
blockmove_discrete = (x, u) -> x + h * blockmove_continuous(x, u)

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

fname = quasi_newton ? "examples/results/blockmove_QN.txt" : "examples/results/blockmove.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (s)   solver(s)  \n")
    for seed = 1:1
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
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %5.1f       %5.1f  \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf, wall_time * 1000, solver_time * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        end
    end
end

# ## Plot solution

x_sol, u_sol = get_trajectory(solver)
    
x = map(x -> x[1], x_sol)
v = map(x -> x[2], x_sol)
s1 = map(u -> u[2], u_sol)
s2 = map(u -> u[3], u_sol)
F = map(u -> u[1], u_sol)
work = [abs(vk * Fk) for (vk, Fk) in zip(v, F)]
s = [s1k - s2k for (s1k, s2k) in zip(s1, s2)]

using DelimitedFiles
open("ipddp_bm_xv.txt", "w") do io
    writedlm(io, [x v])
end

open("ipddp_bm_Fw.txt", "w") do io
    writedlm(io, [F work s])
end

# if visualise
#     x_sol, u_sol = get_trajectory(solver)
    
#     x = map(x -> x[1], x_sol)
#     v = map(x -> x[2], x_sol)
#     u = [map(u -> u[1], u_sol); 0.0]
#     work = [abs(vk * uk) for (vk, uk) in zip(v, u)]
#     plot(range(0, (N-1) * h, length=N), [x v u work], label=["x" "v" "u" "work"])
#     savefig("examples/plots/blockmove.png")
    
#     println("Total absolute work: ", sum(work))
# end
