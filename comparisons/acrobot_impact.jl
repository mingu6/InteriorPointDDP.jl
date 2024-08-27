using IterativeLQR
using LinearAlgebra
using Random
using Plots
using MeshCat

h = 0.05
N = 101
visualise = true

options = Options()
options.scaling_penalty = 1.0
options.initial_constraint_penalty = 1e-2
options.max_iterations = 1000
options.max_dual_updates = 9
options.objective_tolerance = 1e-4
options.lagrangian_gradient_tolerance = 1e-4

Random.seed!(0)

include("../examples/models/acrobot.jl")

if visualise
	include("../examples/visualise/visualise_acrobot.jl")
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

function objk(x, u)
	J = 0.0 

	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)]
	v1 = (q2 - q1) ./ h

	J += 0.1 * h * transpose(v1) * v1
	J += 0.5 * h * u[1] * u[1]
	return J
end

function objN(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 100.0 *  dot(v1, v1)
    J += 400.0 * dot(q2 - qN, q2 - qN)
	return J
end

stage = Cost(objk, nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost(objN, nx, 0),
]

# ## Constraints - perturb complementarity to make easier

stage_constr = Constraint((x, u) -> [
            implicit_contact_dynamics(acrobot_impact, x, u, h, 0.0);
            -u[nq+2:nq+2+2*nc:end]
            ],
            nx, ny)

constraints = [[stage_constr for k = 1:N-1]...,
                Constraint()
                # Constraint((x, u) -> x - xN, nx, 0)
                ]

# ## Initialise solver and solve

solver = Solver(dynamics, objective, constraints; options=options)
q2_init = LinRange(q1, qN, N)[2:end]
ū = [[1.0e-3 * randn(nu); q2_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]

x̄ = rollout(dynamics, x0, ū)

initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

solve!(solver)

# ## Plot solution

x_mat = reduce(vcat, transpose.(solver.problem.nominal_states))
q1 = x_mat[:, 1]
q2 = x_mat[:, 2]
v1 = (x_mat[:, 3] - x_mat[:, 1]) ./ h
v2 = (x_mat[:, 4] - x_mat[:, 2]) ./ h
u_mat = [map(x -> x[1], solver.problem.nominal_actions[1:end-1]); 0.0]
plot(range(0, (N-1) * h, length=N), [q1 q2 v1 v2 u_mat], label=["q1" "q2" "v1" "v2" "u"])
savefig("comparisons/plots/acrobot_impact.png")

if visualise
	q_sol = state_to_configuration(solver.problem.nominal_states)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end
