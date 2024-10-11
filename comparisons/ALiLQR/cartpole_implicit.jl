using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using MeshCat

h = 0.05
N = 101
visualise = true

options = Options()
options.scaling_penalty = 1.2
options.initial_constraint_penalty = 1e-3
options.max_iterations = 1000
options.max_dual_updates = 50
options.objective_tolerance = 1e-5
options.lagrangian_gradient_tolerance = 1e-5

Random.seed!(0)

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

x1 = [0.0; 0.0; 0.0; 0.0]
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

stage_constr = Constraint((x, y) -> [y[1] - 4.0; -y[1] - 4.0; implicit_dynamics(cartpole, x, y) * h],
                    nx, ny, indices_inequality=collect(1:2))

constraints = [stage_constr for k = 1:N-1]

constraints = [[stage_constr for k = 1:N-1]..., Constraint()]

# ## Initialise solver and solve

solver = Solver(dynamics, objective, constraints; options=options)
ȳ = [1.0e-2 * (rand(ny) .- 0.5) for k = 1:N-1]

x̄ = rollout(dynamics, x1, ȳ)

initialize_controls!(solver, ȳ)
initialize_states!(solver, x̄)

solve!(solver)

# ## Visualise solution

x_sol, y_sol = get_trajectory(solver)

if visualise
	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end

# ## Benchmark solver

using BenchmarkTools
solver.options.verbose = false
info = @benchmark solve!($solver) setup=(initialize_controls!($solver, $ȳ), initialize_states!($solver, $x̄))
