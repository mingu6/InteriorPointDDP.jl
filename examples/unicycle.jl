# PREAMBLE

# PKG_SETUP

# ## Setup

using InteriorPointDDP
using LinearAlgebra
using Plots

# ## horizon 
T = 400

# ## acrobot 
num_state = 3 
num_action = 1 

function unicycle_discrete(x, u)
    h = 0.02
    v = 1.5
    φ = x[3]
    return [x[1] + h * v * cos(φ), x[2] + h * v * sin(φ), φ + h * u[1]]
end

# ## model
unicycle = Dynamics(unicycle_discrete, num_state, num_action)
dynamics = [unicycle for t = 1:T-1] ## best to instantiate acrobot once to reduce codegen overhead

# ## initialization
x1 = [-10.0; 0.0; 0.07] 
# ū = [0.01 * zeros(num_action) for t = 1:T-1]
ū = [0.02 * rand(num_action) .- 0.01 for t = 1:T-1]
# ū = [0.1 * ones(num_action) for t = 1:T-1]
x̄ = rollout(dynamics, x1, ū)

stage = Cost((x, u) -> 0.001 * dot(x, x) + 0.0001 * dot(u, u), num_state, num_action)
objective = [
    [stage for t = 1:T-1]...,
    Cost((x, u) -> 0.001 * dot(x, x), num_state, 0),
] 

stage_constr = Constraint((x, u) -> [
    u[1] - 1.5; 
    -u[1] - 1.5;
    x[2] - 1.0;
    -x[2] - 1.0;
    1.0 - norm([x[1] + 5.5, x[2] + 1], 2);
    0.5 - norm([x[1] + 8.0, x[2] - 0.2], 2);
    1.5 - norm([x[1] + 2.5, x[2] - 1.0], 2)
], 
num_state, num_action, indices_inequality=collect(1:7))

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
