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

include("models/acrobot.jl")

if visualise
	include("visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = acrobot_impact.nq
nc = acrobot_impact.nc
nu = acrobot_impact.nu
nx = 2 * nq
ny = nu + nq + 2 * nc

q1 = T[0.0; 0.0]
q2 = T[0.0; 0.0]
x1 = T[q1; q2]
qN = T[π; 0.0]
xN = T[qN; qN]

# ## Dynamics - implicit variational integrator (midpoint)

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nu .+ (1:nq)]], nx, ny)
dynamics = [dyn_acrobot for k = 1:N-1]

# ## Costs

function objt(x, u)
	J = 0.0 

	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)]
	v1 = (q2 - q1) ./ h

	J += 0.01 * h * transpose(v1) * v1
	J += 0.01 * h * u[1] * u[1]
	return J
end

function objT(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 100.0 *  dot(v1, v1)
    J += 500.0 * dot(q2 - qN, q2 - qN)
	return J
end

stage = Cost(objt, nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost(objT, nx, 0),
]

# ## Constraints

stage_constr = Constraint((x, u) -> implicit_contact_dynamics(acrobot_impact, x, u, h),
			nx, ny, indices_compl=[5, 6])

constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(
	[-T(Inf); -T(Inf) * ones(T, nq); zeros(T, nc); zeros(T, nc)],
	[T(Inf); T(Inf) * ones(T, nq); T(Inf) * ones(T, nc); T(Inf) * ones(T, nc)]
)
bounds = [bound for k in 1:N-1]

# ## Initialise solver and solve

q2_init = LinRange(q1, qN, N)[2:end]
ū = [[T(1.0e-3) * randn(T, nu); q2_init[k]; T(0.01) * ones(T, nc); T(0.01) * ones(T, nc)] for k = 1:N-1]
solver = Solver(T, dynamics, objective, constraints, bounds, options=options)
solve!(solver, x1, ū)

# ## Plot solution

x_sol, u_sol = get_trajectory(solver)
x_mat = reduce(vcat, transpose.(x_sol))
q1 = x_mat[:, 1]
q2 = x_mat[:, 2]
v1 = (x_mat[:, 3] - x_mat[:, 1]) ./ h
v2 = (x_mat[:, 4] - x_mat[:, 2]) ./ h
u_mat = [map(x -> x[1], u_sol); 0.0]
λ1 = [map(x -> x[end-1], u_sol); 0.0]
λ2 = [map(x -> x[end], u_sol); 0.0]
# plot(range(0, (N-1) * h, length=N), [q1 q2 v1 v2 u_mat λ1 λ2], label=["q1" "q2" "v1" "v2" "u" "λ1" "λ2"])
# plot(range(0, (N-1) * h, length=N), [v1 v2 λ1 λ2], label=["v1" "v2" "λ1" "λ2"])
# plot(range(0, (N-1) * h, length=N), [λ1 λ2], label=["λ1" "λ2"])
# plot(range(0, (N-1) * h, length=N), [q1 q2], label=["q1" "q2"])
plot(range(0, (N-1) * h, length=N), [q2 λ1 λ2 v2], label=["q2" "λ1" "λ2" "v2"])
savefig("examples/plots/acrobot_impact.png")

if visualise
	q_sol = state_to_configuration(x_sol)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end

# ## Benchmark solver

using BenchmarkTools
solver.options.verbose = false
@benchmark solve!(solver, x1, ū)
