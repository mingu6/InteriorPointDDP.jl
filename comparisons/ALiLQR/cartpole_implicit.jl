using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using MeshCat
using BenchmarkTools
using Printf

visualise = false
benchmark = true
verbose = true

h = 0.05
N = 101

options = Options()
options.scaling_penalty = 1.3
options.initial_constraint_penalty = 1e-3
options.max_dual_updates = 50

include("../../examples/models/cartpole.jl")

if visualise
	include("../../examples/visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = cartpole.nq
nu = cartpole.nu
nx = 2 * nq
ny = nu + nq  # torque and acceleration now decision variables/"controls"

xN = [0.0; π; 0.0; 0.0]

# ## Dynamics - implicit dynamics with RK2 integration

f = (x, y) -> [x[nq .+ (1:nq)]; y[nu .+ (1:nq)]]
cartpole_discrete = (x, y) -> x + h * f(x + 0.5 * h * f(x, y), y)  # Explicit midpoint
cartpole_dyn = Dynamics(cartpole_discrete, nx, ny)
dynamics = [cartpole_dyn for k = 1:N-1]

# ## Costs

stage = Cost((x, y) -> h * dot(y[1], y[1]), nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, y) -> 400. * dot(x - xN, x - xN), nx, 0)
] 

# ## Constraints

stage_constr = Constraint((x, y) -> [y[1] - 4.0; -y[1] - 4.0; implicit_dynamics(cartpole, x, y)],
                    nx, ny, indices_inequality=collect(1:2))

constraints = [stage_constr for k = 1:N-1]

constraints = [[stage_constr for k = 1:N-1]..., Constraint()]

solver = Solver(dynamics, objective, constraints; options=options)

open("results/cartpole_implicit.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms)  \n")
	for seed = 1:50
		solver.options.verbose = verbose
		Random.seed!(seed)
		
		# ## Initialise solver and solve
		
		x1 = [0.0; 0.0; 0.0; 0.0] + (rand(4) .- 0.5) .* [0.05, 0.2, 0.1, 0.1]
		ȳ = [1.0e-2 * (rand(ny) .- 0.5) for k = 1:N-1]

		x̄ = rollout(dynamics, x1, ȳ)
		solve!(solver, x̄, ȳ)
        
        if benchmark
            solver.options.verbose = false
            solve_time = @belapsed solve!($solver, $x̄, $ȳ) samples=10
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %5.1f       %5.1f\n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1],
                            solver.data.max_violation[1], solve_time * 1000, 0.0)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.iterations[1], solver.data.status[1], solver.data.objective[1], solver.data.max_violation[1])
        end
    end
end

# ## Visualise solution

if visualise
	x_sol, y_sol = get_trajectory(solver)

	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end
