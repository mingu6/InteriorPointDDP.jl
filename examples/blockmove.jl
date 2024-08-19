using InteriorPointDDP
using LinearAlgebra
using Random
using Plots

dt = 0.01
N = 101
seed = 1
xN = [1.0; 0.0]
x1 = [0.0; 0.0]
options = Options(quasi_newton=false, verbose=true, max_iterations=1500, optimality_tolerance=1e-7)

num_state = 2  # position and velocity
num_action = 3  # pushing force, 2x slacks for + and - components of abs work

function blockmove_continuous(x, u)
    return [x[2], u[1]]
end

# explicit midpoint for integrator
blockmove_discrete = (x, u) -> x + dt * blockmove_continuous(x + 0.5 * dt * blockmove_continuous(x, u), u)

blockmove_dyn = Dynamics(blockmove_discrete, 2, 3)
dynamics = [blockmove_dyn for k = 1:N-1]

stage_cost = Cost((x, u) -> dt * (u[2] + u[3]), 2, 3)
objective = [
    [stage_cost for k = 1:N-1]...,
    Cost((x, u) -> 400.0 * dot(x - xN, x - xN), 2, 0),
]

stage_constr = Constraint((x, u) -> [
    u[2] - u[3] - u[1] * x[2]
], 
2, 3, 
bounds_lower=[-10.0, 0.0, 0.0], bounds_upper=[10.0, Inf, Inf])
constraints = [stage_constr for k = 1:N-1]

Random.seed!(seed)
s_init = 0.01 * ones(2)
ū = [[1.0e-3 * randn(1); deepcopy(s_init)] for t = 1:N-1]
solver = Solver(dynamics, objective, constraints, options=options)
solve!(solver, x1, ū)

# plot solution
x_mat = reduce(vcat, transpose.(solver.problem.nominal_states))
x = x_mat[:, 1]
v = x_mat[:, 2]
u = [map(u -> u[1], solver.problem.nominal_actions[1:end-1]); 0.0]
sp = [map(u -> u[2], solver.problem.nominal_actions[1:end-1]); 0.0]
sn = [map(u -> u[3], solver.problem.nominal_actions[1:end-1]); 0.0]
plot(range(0, (N-1) * dt, length=N), [x v u sp sn], label=["x" "v" "u" "s+" "s-"])
savefig("examples/plots/blockmove.png")
