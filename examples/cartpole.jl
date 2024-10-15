using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat

T = Float64
h = 0.05
N = 101
options = Options{T}(quasi_newton=false, verbose=true)
visualise = true

Random.seed!(0)
include("models/cartpole.jl")

if visualise
	include("visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = cartpole.nq
nu = cartpole.nu
nx = 2 * nq

# ## Dynamics

x1 = T[0.0; 0.0; 0.0; 0.0]
xN = T[0.0; π; 0.0; 0.0]

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

stage_constr = Constraint((x, u) -> [], nx, nu)

constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(
	-T(4.0) * ones(T, nu),
	T(4.0) * ones(T, nu)
)
# bound = Bound(T, nu)
bounds = [bound for k in 1:N-1]

# ## Initialise solver and solve

ū = [T(1.0e-2) * (rand(T, nu) .- 0.5) for k = 1:N-1]
solver = Solver(T, dynamics, objective, constraints, bounds, options=options)
solve!(solver, x1, ū)

x_sol, u_sol = get_trajectory(solver)

if visualise
	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end

# ## Benchmark solver

using BenchmarkTools
solver.options.verbose = false
@benchmark solve!($solver, $x1, $ū)
