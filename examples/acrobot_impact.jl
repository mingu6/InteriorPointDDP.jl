using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat

h = 0.05
N = 101
options = Options(quasi_newton=false, verbose=true, max_iterations=5000, optimality_tolerance=1e-5)
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

q1 = [0.0; 0.0]
q2 = [0.0; 0.0]
x0 = [q1; q2]
qN = [π; 0.0]
xN = [qN; qN]

# ## Dynamics - implicit variational integrator (midpoint)

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nu .+ (1:nq)]], nx, ny)
dynamics = [dyn_acrobot for k = 1:N-1]

# ## Costs

function objt(x, u)
	J = 0.0 

	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)]
	v1 = (q2 - q1) ./ h

	J += 0.2 * h * transpose(v1) * v1
	J += 0.5 * h * u[1] * u[1]
	J += 0.2 * h * dot(q2 - qN, q2 - qN)
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
			nx, ny,
            bounds_lower=[-Inf; -Inf * ones(nq); zeros(nc); zeros(nc)],
            bounds_upper=[Inf; Inf * ones(nq); Inf * ones(nc); Inf * ones(nc)],
			indices_compl=[5, 6])

constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(
	[-Inf; -Inf * ones(nq); zeros(nc); zeros(nc)],
	[Inf; Inf * ones(nq); Inf * ones(nc); Inf * ones(nc)]
)
bounds = [bound for k in 1:N-1]

# ## Initialise solver and solve

q2_init = LinRange(q1, qN, N)[2:end]
ū = [[1.0e-3 * randn(nu); q2_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
solver = Solver(dynamics, objective, constraints, bounds, options=options)
solve!(solver, x0, ū)

# ## Plot solution

x_mat = reduce(vcat, transpose.(solver.problem.nominal_states))
q1 = x_mat[:, 1]
q2 = x_mat[:, 2]
v1 = (x_mat[:, 3] - x_mat[:, 1]) ./ h
v2 = (x_mat[:, 4] - x_mat[:, 2]) ./ h
u_mat = [map(x -> x[1], solver.problem.nominal_actions[1:end-1]); 0.0]
plot(range(0, (N-1) * h, length=N), [q1 q2 v1 v2 u_mat], label=["q1" "q2" "v1" "v2" "u"])
savefig("examples/plots/acrobot_impact.png")

if visualise
	q_sol = state_to_configuration(solver.problem.nominal_states)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end

# ## Benchmark solver

using BenchmarkTools
solver.options.verbose = false
@benchmark solve!(solver, x0, ū)
