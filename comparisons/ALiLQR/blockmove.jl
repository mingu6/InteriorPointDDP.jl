using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using BenchmarkTools
using Printf

visualise = true
benchmark = true
verbose = true

N = 101
h = 0.01
xN = [1.0; 0.0]
x0 = [0.0; 0.0]

options = Options()
options.scaling_penalty = 1.0
options.initial_constraint_penalty = 1e-4

num_state = 2
num_control = 3  # force and two slack variables to represent abs work

# ## Dynamics - explicit midpoint for integrator

function blockmove_continuous(x, u)
    return [x[2], u[1]]
end
blockmove_discrete = (x, u) -> x + h * blockmove_continuous(x + 0.5 * h * blockmove_continuous(x, u), u)

blockmove_dyn = Dynamics(blockmove_discrete, num_state, num_control)
dynamics = [blockmove_dyn for k = 1:N-1]

# ## Costs

stage = Cost((x, u) -> h * (u[2] + u[3]), num_state, num_control)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, u) -> 400.0 * dot(x - xN, x - xN), num_state, 0),
]

# ## Constraints

stage_constr = Constraint((x, u) -> [
    u[2] - u[3] - u[1] * x[2],
    - u[1] - 10.0,
    -u[2],
    -u[3],
    u[1] - 10.0 
    ], 
num_state, num_control, indices_inequality=collect(2:4))

constraints = [
[stage_constr for k = 1:N-1]..., 
    Constraint()
]

# ## Initialise solver and solve

solver = Solver(dynamics, objective, constraints; options=options)

open("results/blockmove.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        time (s)  \n")
	for seed = 1:50
		solver.options.verbose = verbose
		Random.seed!(seed)
        ū = [[1.0e-0 * (rand(1) .- 0.5); -0.01 * ones(2)] for k = 1:N-1]
        x̄ = rollout(dynamics, x0, ū)
        
        solve!(solver, x̄, ū)
		
		if benchmark
            solver.options.verbose = false
            solve_time = @belapsed solve!($solver, $x̄, $ū)
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %.5f  \n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1], solver.data.max_violation[1], solve_time)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1], solver.data.max_violation[1])
        end
    end
end

# ## Plot solution

if visualise
    x = map(x -> x[1], solver.problem.nominal_states)
    v = map(x -> x[2], solver.problem.nominal_states)
    u = [map(u -> u[1], solver.problem.nominal_actions[1:end-1]); 0.0]
    work = [abs(vk * uk) for (vk, uk) in zip(v, u)]
    plot(range(0, (N-1) * h, length=N), [x v u work], label=["x" "v" "u" "work"])
    savefig("plots/blockmove.png")
    
    println("Total absolute work: ", sum(work))
end
