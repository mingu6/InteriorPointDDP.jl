using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat

dt = 0.1
N = 26
seed = 1
options = Options(quasi_newton=true, verbose=true, max_iterations=12, optimality_tolerance=1e-5)

include("models/planar_push.jl")
include("visualise/visualise_pp.jl")

# vis = Visualizer() 
# render(vis)

# ## mode
MODE = :translate
# MODE = :rotate 

nq = planarpush.nq
nc = planarpush.nc
nc_impact = 1
nu = planarpush.nu
nx = 2 * nq
ny = nu + nq + 2 * nc_impact + 5 + 9 + 5 + 9

x1 = zeros(2 * nq)
xT = zeros(2 * nq)

r_dim = 0.1
if MODE == :translate 
	q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
	q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
    x1 .= [q1; q1]
	x_goal = 1.0
	y_goal = 0.0
	θ_goal = 0.0 * π
	qT = [x_goal, y_goal, θ_goal, x_goal - r_dim, y_goal - r_dim]
	xT .= [qT; qT]
elseif MODE == :rotate 
	q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
	q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
	x1 .= [q1; q1]
	x_goal = 0.5
	y_goal = 0.5
	θ_goal = 0.5 * π
	qT = [x_goal, y_goal, θ_goal, x_goal-r_dim, y_goal-r_dim]
	xT .= [qT; qT]
end

dyn_pp = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nu .+ (1:nq)]], nx, ny)
dynamics = [dyn_pp for k = 1:N-1]


# ## objective
function objt(x, u)
	J = 0.0 

	q1 = x[1:planarpush.nq]
	q2 = x[planarpush.nq .+ (1:planarpush.nq)] 
	v1 = (q2 - q1) ./ dt

	J += 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT) 
	J += 0.5 * (MODE == :translate ? 1.0e-1 : 1.0e-2) * transpose(u) * u

	return J
end


function objT(x, u)
	J = 0.0 
	
	q1 = x[1:planarpush.nq] 
	q2 = x[planarpush.nq .+ (1:planarpush.nq)] 
	v1 = (q2 - q1) ./ dt

	J += 5.0 * 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1
	J += 10.0 * 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT)

	return J
end


stage = Cost(objt, nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost(objT, nx, 0),
]

stage_constr = Constraint((x, u) -> implicit_contact_dynamics(planarpush, x, u, dt), nx, ny,
            bounds_lower=[-10.0; -10.0; -Inf * ones(nq); zeros(2 * nc_impact); -Inf * ones(5); -Inf * ones(9); -Inf * ones(5); -Inf * ones(9)],
            bounds_upper=[10.0; 10.0; Inf * ones(nq); Inf * ones(2 * nc_impact); Inf * ones(5); Inf * ones(9); Inf * ones(5); Inf * ones(9)],
			indices_compl=[21, 22, 25, 28, 31, 34])

constraints = [stage_constr for k = 1:N-1]

Random.seed!(seed)
q2_init = LinRange(q1, qT, N)[2:end]
u_init = MODE == :translate ? [t < 5 ? [1.0; 0.0] : [0.0; 0.0] for t = 1:N-1] : [t < 5 ? [1.0; 0.0] : t < 10 ? [0.5; 0.0] : [0.0; 0.0] for t = 1:N-1]
mc = 1e-1
ms = 1e-2
ū = [[u_init[t]; q2_init[t]; ms * ones(2); mc * ones(5); ms * ones(9); mc * ones(5); ms * ones(9)] for t = 1:N-1]
solver = Solver(dynamics, objective, constraints, options=options)
solve!(solver, x1, ū)

q_sol = state_to_configuration(solver.problem.nominal_states)
visualize!(vis, planarpush, q_sol, Δt=dt);
