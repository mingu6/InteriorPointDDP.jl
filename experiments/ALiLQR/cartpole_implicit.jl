using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using MeshCat
using BenchmarkTools
using Printf

visualise = false
benchmark = false
verbose = true

h = 0.05
N = 101

options = Options()
options.scaling_penalty = 1.3
options.initial_constraint_penalty = 1e-3
options.max_dual_updates = 50

include("../models/cartpole.jl")

if visualise
	include("../visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = cartpole.nq
nF = cartpole.nu
nx = 2 * nq
nu = nF + nq  # torque and acceleration now decision variables/"controls"

xN = [0.0; π; 0.0; 0.0]

# ## Dynamics - forward Euler

f = (x, u) -> x + h * [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]  # forward Euler
cartpole_dyn = Dynamics(f, nx, nu)
dynamics = [cartpole_dyn for k = 1:N-1]

# ## Costs

stage = Cost((x, u) -> h * dot(u[1], u[1]), nx, nu)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, u) -> 400. * dot(x - xN, x - xN), nx, 0)
] 

# ## Constraints

stage_constr = Constraint((x, u) -> [u[1] - 4.0; -u[1] - 4.0; implicit_dynamics(cartpole, x, u)],
                    nx, nu, indices_inequality=collect(1:2))

constraints = [stage_constr for k = 1:N-1]

constraints = [[stage_constr for k = 1:N-1]..., Constraint()]

# ## Initialise solver
		
solver = Solver(dynamics, objective, constraints; options=options)

open("results/cartpole_implicit.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms)  \n")
	for seed = 1:50
		solver = Solver(dynamics, objective, constraints; options=options)
		solver.options.verbose = verbose
		
		Random.seed!(seed)
		x1 = (rand(4) .- 0.5) .* [0.1, 0.1, 0.1, 0.1]
		ū = [1.0e-1 * (rand(nu) .- 0.5) for k = 1:N-1]
		x̄ = rollout(dynamics, x1, ū)
		
		solve!(solver, x̄, ū)
        
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

# ## Visualise solution

if visualise
	x_sol, u_sol = get_trajectory(solver)

	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end
