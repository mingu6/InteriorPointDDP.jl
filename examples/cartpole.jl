# PREAMBLE

# PKG_SETUP

# ## Setup

using InteriorPointDDP
using LinearAlgebra
using Plots

# ## horizon 
T = 51 

# ## acrobot 
num_state = 4 
num_action = 1 

function cartpole_continuous(x, u)
    mc = 1.0  # mass of the cart in kg (10)
    mp = 0.2   # mass of the pole (point mass at the end) in kg
    l = 0.5   # length of the pole in m
    g = 9.81  # gravity m/s^2

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

function cartpole_discrete(x, u)
    h = 0.1 # timestep 
    x + h * cartpole_continuous(x + 0.5 * h * cartpole_continuous(x, u), u)
end

# ## model
cartpole = Dynamics(cartpole_discrete, num_state, num_action)
dynamics = [cartpole for t = 1:T-1] ## best to instantiate acrobot once to reduce codegen overhead

# ## initialization
x1 = [0.0; 0.0; 0.0; 0.0] 
xT = [0.0; π; 0.0; 0.0]
ū = [0.01 * ones(num_action) for t = 1:T-1]
x̄ = rollout(dynamics, x1, ū)

stage = Cost((x, u) -> 0.01 * dot(x - xT, x - xT) + 0.1 * dot(u, u), num_state, num_action)
objective = [
    [stage for t = 1:T-1]...,
    Cost((x, u) -> 100.0 * dot(x - xT, x - xT), num_state, 0),
] 

stage_constr = Constraint((x, u) -> [
    u[1] - 5.0; 
    -u[1] - 5.0
], 
num_state, num_action, indices_inequality=collect(1:2))

constraints = [
    [stage_constr for t = 1:T-1]..., 
        Constraint() # no terminal constraint 
]
# constraints = [
#     [Constraint() for t = 1:T-1]..., 
#         Constraint() # no terminal constraint 
# ]

# ## solver
solver = Solver(dynamics, objective, constraints)
# solver = Solver(dynamics, objective)
initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

# ## visuals
plot(hcat(x_sol...)')
# plot(hcat(u_sol...)', linetype=:steppost)

# # ## benchmark allocations + timing
# using BenchmarkTools
# info = @benchmark solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))
# display(info)

# function profile_solve(x̄, ū, n)
#     for i = 1:n
#         initialize_controls!(solver, ū)
#         initialize_states!(solver, x̄)
#         solve!(solver)
#     end
# end

# using ProfileView
# @profview profile_solve(x̄, ū, 1)  # run once to trigger compilation (ignore this one)
# @profview profile_solve(x̄, ū, 5)
