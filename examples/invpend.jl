# PREAMBLE

# PKG_SETUP

# ## Setup

using InteriorPointDDP
using LinearAlgebra
using Plots

# ## horizon 
# NOTE: This should be one more than the matlab horizon
T = 501

# ## inverse pendulum 
num_state = 2
num_action = 1
num_parameter = 0
h = 0.05

function invpend_dynamics(x, u)
    h = 0.05
    return [
        x[1] .+ h * x[2];
        x[2] .+ h * sin(x[1]) .+ h * u;
    ]
end

# ## model
invpend = Dynamics(invpend_dynamics, num_state, num_action)
dynamics = [invpend for t = 1:T-1] 

# ## initialization
x1 = [-pi; 0.0]

# ## rollout
ū = [0.01 * ones(num_action) for t = 1:T-1]
x̄ = rollout(dynamics, x1, ū)

# ## objective 

# ## Define aux functions for stage cost and final cost 
function stage_cost(x, u)
    Q = I(num_state)
    R = I(num_action)
    return 0.5 * h * (transpose(x) * Q * x + transpose(u) * R * u)
end

function final_cost(x)
    P = 10 * I(num_state)
    return 0.5 * transpose(x) * P * x
end

objective = [
    [Cost((x, u) -> stage_cost(x,u), num_state, num_action) for t = 1:T-1]...,
    Cost((x, u) -> final_cost(x), num_state, 0)
]

# ## constraints
constraints = [
    [Constraint((x, u) -> [
            u[1] - 0.25; 
            -u[1] - 0.25
        ], 
        num_state, num_action, indices_inequality=collect(1:2)) for t = 1:T-1]..., 
        Constraint() # no terminal constraint 
]

# ## solver
solver = Solver(dynamics, objective, constraints)
initialize_controls!(solver, ū)
initialize_states!(solver, x̄)

## solve
solve!(solver)

## solution
x_sol, u_sol = get_trajectory(solver)

## visualize
plot(hcat(x_sol...)')
plot(hcat(u_sol[1:end-1]...)', linetype=:steppost)

## benchmark allocations + timing
# using BenchmarkTools
# info = @benchmark solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))
# display(info)
