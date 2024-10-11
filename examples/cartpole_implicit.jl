using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat

T = Float64
h = 0.05
N = 101
options = Options{T}(quasi_newton=false, verbose=true, max_iterations=5000, optimality_tolerance=1e-5)
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
ny = nu + nq  # torque and acceleration now decision variables/"controls"

x1 = T[0.0; 0.0; 0.0; 0.0]
xN = T[0.0; π; 0.0; 0.0]

# ## Dynamics - implicit dynamics with RK2 integration

f = (x, y) -> [x[nq .+ (1:nq)]; y[nu .+ (1:nq)]]
cartpole_discrete = (x, y) -> x + h * f(x + 0.5 * h * f(x, y), y)  # Explicit midpoint
cartpole_dyn = Dynamics(cartpole_discrete, nx, ny)
dynamics = [cartpole_dyn for k = 1:N-1]

# ## Costs

stage = Cost((x, u) -> h * dot(u[1], u[1]), nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, u) -> 400. * dot(x - xN, x - xN), nx, 0)
] 

# ## Constraints

stage_constr = Constraint((x, y) -> implicit_dynamics(cartpole, x, y) * h, nx, ny)

constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(
	[-T(5.0) * ones(T, nu); -T(Inf) * ones(T, nq)],
	[T(5.0) * ones(T, nu); T(Inf) * ones(T, nq)]
)
bounds = [bound for k in 1:N-1]


# ## Initialise solver and solve

ū = [T(1.0e-2) * (rand(T, ny) .- 0.5) for k = 1:N-1]
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

