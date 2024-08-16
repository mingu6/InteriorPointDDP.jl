using InteriorPointDDP
using LinearAlgebra
using Symbolics
using Random
using Plots
using MeshCat

h = 0.05
N = 101
seed = 1
options = Options(quasi_newton=false, verbose=true, max_iterations=701, optimality_tolerance=1e-5)

include("models/acrobot_impact.jl")
include("visualise/visualise_acrobot.jl")

# vis = Visualizer() 
# render(vis)

nq = acrobot_impact.nq
nc = acrobot_impact.nc
nu = acrobot_impact.nu
nx = 2 * nq
ny = nu + nq + 2 * nc

q1 = [0.0; 0.0]
q2 = [0.0; 0.0]
x1 = [q1; q2]
qN = [π; 0.0]
xN = [qN; qN]

dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nu .+ (1:nq)]], nx, ny)
dynamics = [dyn_acrobot for k = 1:N-1]

function objt(x, u)
	J = 0.0 

	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)]
	v1 = (q2 - q1) ./ h

	J += 0.1 * h * transpose(v1) * v1
	J += 0.1 * h * u[1] * u[1]
	return J
end

function objT(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 50.0 * dot(v1, v1)
    J += 200.0 * dot(q2 - qN, q2 - qN)
	return J
end

stage = Cost(objt, nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost(objT, nx, 0),
]

stage_constr = Constraint(implicit_contact_dynamics, nx, ny,
            bounds_lower=[-Inf; -Inf * ones(nq); zeros(nc); zeros(nc)],
            bounds_upper=[Inf; Inf * ones(nq); Inf * ones(nc); Inf * ones(nc)])

constraints = [stage_constr for k = 1:N-1]

Random.seed!(seed)
sλ_init = 0.01 * ones(nc)
q2_init = LinRange(q1, qN, N)[2:end]
ū = [[1.0e-3 * randn(nu); q2_init[t]; deepcopy(sλ_init); deepcopy(sλ_init)] for t = 1:N-1]
# ū = [1.0e-3 * randn(ny) for t = 1:N-1]
solver = Solver(dynamics, objective, constraints, options=options)
solve!(solver, x1, ū)

# plot solution
x_mat = reduce(vcat, transpose.(solver.problem.nominal_states))
q1 = x_mat[:, 1]
q2 = x_mat[:, 2]
v1 = (x_mat[:, 3] - x_mat[:, 1]) ./ h
v2 = (x_mat[:, 4] - x_mat[:, 2]) ./ h
u_mat = [map(x -> x[1], solver.problem.nominal_actions[1:end-1]); 0.0]
plot(range(0, (N-1) * 0.05, length=N), [q1 q2 v1 v2 u_mat], label=["q1" "q2" "v1" "v2" "u"])
savefig("examples/plots/acrobot_impact.png")

q_sol = state_to_configuration(solver.problem.nominal_states)
visualize!(vis, acrobot_impact, q_sol, Δt=h);
