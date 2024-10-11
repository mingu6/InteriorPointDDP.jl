using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using MeshCat

h = 0.05
N = 101
visualise = true

options = Options()
options.scaling_penalty = 1.5
options.initial_constraint_penalty = 1e-2
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

x1 = [0.0; 0.0; 0.0; 0.0]
xN = [0.0; π; 0.0; 0.0]

# ## Dynamics - implicit dynamics with RK2 integration

cartpole_continuous = (x, u) -> [x[nq .+ (1:nq)]; forward_dynamics(cartpole, x, u)]
cartpole_discrete = (x, u) -> x + h * cartpole_continuous(x + 0.5 * h * cartpole_continuous(x, u), u)
cartpole_dyn = Dynamics(cartpole_discrete, nx, nu)
dynamics = [cartpole_dyn for k = 1:N-1]

# ## Costs

stage = Cost((x, u) -> h * dot(u[1], u[1]), nx, nu)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, u) -> 400. * dot(x - xN, x - xN), nx, 0)
] 

# ## Constraints

stage_constr = Constraint((x, u) -> [u[1] - 4.0; -u[1] - 4.0], nx, nu, indices_inequality=collect(1:2))

constraints = [[stage_constr for k = 1:N-1]..., Constraint()]

# ## Initialise solver and solve

solver = Solver(dynamics, objective, constraints; options=options)
ū = [1.0e-2 * (rand(nu) .- 0.5) for k = 1:N-1]

x̄ = rollout(dynamics, x1, ū)

initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

solve!(solver)

# ## Visualise solution

x_sol, u_sol = get_trajectory(solver)

if visualise
	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end

# ## Benchmark solver

# using BenchmarkTools
# solver.options.verbose = false
# info = @benchmark solve!($solver) setup=(initialize_controls!($solver, $ū), initialize_states!($solver, $x̄))
