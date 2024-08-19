using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat

N = 101
dt = 0.05
num_state = 4 
num_action = 2 # one is linear force, one is slack for state ineq. constraint
xN = [0.0; π; 0.0; 0.0]
x1 = zeros(4)
options = Options(quasi_newton=false, verbose=true, max_iterations=1500, optimality_tolerance=1e-5)

include("models/cartpole.jl")
include("visualise/visualise_cartpole.jl")

# vis = Visualizer() 
# render(vis)

function cartpole_continuous(model::Cartpole{T}, x, u) where T
    mc = model.mc
    mp = model.mp
    l = model.l
    g = model.g

    q = x[[1,2] ]
    qd = x[[3,4]]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0.0 -mp*qd[2]*l*s; 0 0]
    G = [0.0, mp*g*l*s]
    B = [1.0, 0.0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

cartpole_discrete = (x, u) -> x + dt * cartpole_continuous(cartpole, x + 0.5 * dt * cartpole_continuous(cartpole, x, u), u)
# cartpole_discrete = (x, u) -> x + dt * cartpole_continuous(cartpole, x, u)
cartpole_dyn = Dynamics(cartpole_discrete, num_state, num_action)
dynamics = [cartpole_dyn for k = 1:N-1]

stage = Cost((x, u) -> dt * dot(u[1], u[1]), num_state, num_action)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, u) -> 400. * (sqrt(dot(x - xN, x - xN) + 1e-12) - 1e-6), num_state, 0),
] 

stage_constr = Constraint(
    (x, u) -> [cartpole_discrete(x, u)[1] - u[2]], 
    num_state, num_action, 
    bounds_lower=[-3., -0.10], bounds_upper=[3., 0.2]
)

constraints = [stage_constr for k = 1:N-1]

Random.seed!(seed)
s_init = 0.01 * ones(1)
ū = [[1e-3 .* randn(1); deepcopy(s_init)] for k = 1:N-1]
solver = Solver(dynamics, objective, constraints, options=options)
solve!(solver, x1, ū)

x_mat = reduce(vcat, transpose.(solver.problem.nominal_states))
x = x_mat[:, 1]
θ = x_mat[:, 2]
ẋ = x_mat[:, 3]
θ̇ = x_mat[:, 4]
u = [map(x -> x[1], solver.problem.nominal_actions[1:end-1]); 0.0]
s = [map(x -> x[2], solver.problem.nominal_actions[1:end-1]); 0.0]
plot(range(0, (N-1) * dt, length=N), [x ẋ θ θ̇ u s], label=["x" "ẋ" "θ" "θ̇" "u" "s"])
savefig("examples/plots/cartpole.png")

q_sol = map(x -> [x[1], x[2]], solver.problem.nominal_states)
visualize!(vis, cartpole, q_sol, Δt=dt);
