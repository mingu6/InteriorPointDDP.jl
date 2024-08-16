using InteriorPointDDP
using LinearAlgebra
using Random
using MeshCat

N = 201
dt = 0.05
num_state = 4
num_action = 2 # slack variable for elbow constraint
xN = [π; 0.0; 0.0; 0.0]
multiplier = 1e-1
options = Options(quasi_newton=false, verbose=true, max_iterations=500, optimality_tolerance=1e-6)
seed = 1
x1 = [0.0; 0.0; 0.0; 0.0]

include("models/acrobot_impact.jl")
include("visualise/visualise_acrobot.jl")

# vis = Visualizer() 
# render(vis)

function acrobot_continuous(x, u)
    mass1 = 1.0  
    inertia1 = 0.333  
    length1 = 1.0 
    lengthcom1 = 0.5 

    mass2 = 1.0  
    inertia2 = 0.333  
    length2 = 1.0 
    lengthcom2 = 0.5 

    gravity = 9.81 
    friction1 = 0.1 
    friction2 = 0.1

    function M(x)
        a = (inertia1 + inertia2 + mass2 * length1 * length1
            + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

        b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

        c = inertia2

        return [a b; b c]
    end

    function Minv(x) 
        m = M(x) 
        a = m[1, 1] 
        b = m[1, 2] 
        c = m[2, 1] 
        d = m[2, 2]
        1.0 / (a * d - b * c) * [d -b;-c a]
    end

    function τ(x)
        a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
            - mass2 * gravity * (length1 * sin(x[1])
            + lengthcom2 * sin(x[1] + x[2])))

        b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

        return [a; b]
    end

    function C(x)
        a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
        d = 0.0

        return [a b; c d]
    end

    function B(x)
        [0.0; 1.0]
    end

    q = view(x, 1:2)
    v = view(x, 3:4)

    qdd = Minv(q) * (-1.0 * C(x) * v
            + τ(q) + B(q) * u[1] - [friction1; friction2] .* v)

    return [x[3]; x[4]; qdd[1]; qdd[2]]
end

# acrobot_discrete = (x, u) -> x + dt * acrobot_continuous(x + 0.5 * dt * acrobot_continuous(x, u), u)
acrobot_discrete = (x, u) -> x + dt * acrobot_continuous(x, u)
acrobot_dyn = Dynamics(acrobot_discrete, num_state, num_action)
dynamics = [acrobot_dyn for k = 1:N-1]

stage = Cost((x, u) -> 0.5 * dt * dot(u, u) + dt * 0.5 * dot(x - xN, x - xN), num_state, num_action)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, u) -> 200.0 * dot(x - xN, x - xN), num_state, 0),
] 

stage_constr = Constraint((x, u) -> [
    acrobot_discrete(x, u)[2] + u[2]
], 
num_state, num_action, bounds_lower=[-5.0, -0.5 * π], bounds_upper=[5.0, 0.5 * π])

constraints = [stage_constr for k = 1:N-1]

Random.seed!(seed)
s_init = 0.1 * ones(1)
ū = [[1e-3 .* randn(1); deepcopy(s_init)] for k = 1:N-1]
solver = Solver(dynamics, objective, constraints, options=options)
solve!(solver, x1, ū)

using Plots
x_mat = reduce(vcat, transpose.(solver.problem.nominal_states))
q1 = x_mat[:, 1]
q2 = x_mat[:, 2]
v1 = x_mat[:, 3]
v2 = x_mat[:, 4]
u_mat = [map(x -> x[1], solver.problem.nominal_actions[1:end-1]); 0.0]
plot(range(0, (N-1) * 0.05, length=N), [q1 q2 v1 v2 u_mat], label=["q1" "q2" "v1" "v2" "u"])
savefig("examples/plots/acrobot_state.png")

q_sol = map(x -> [x[1], x[2]], solver.problem.nominal_states)
visualize!(vis, acrobot_impact, q_sol, Δt=dt);