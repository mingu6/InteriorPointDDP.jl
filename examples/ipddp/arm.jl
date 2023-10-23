# PREAMBLE

# PKG_SETUP

# ## Setup

using IterativeLQR 
using LinearAlgebra
using Plots

# ## horizon 
# NOTE: This should be one more than the matlab horizon
T = 51

# ## arm 
num_state = 3 
num_action = 3 
num_parameter = 0 

function arm_dynamics(x, u)
    h = 0.1
    return x .+ h .* u
end

# ## model
arm = Dynamics(arm_dynamics, num_state, num_action) 
dynamics = [arm for t = 1:T-1] 

# ## initialization
x1 = [-pi/2; 0; 0]

# ## rollout
ū = [0.01 * ones(num_action) .- 0.01 for t = 1:T-1]
x̄ = rollout(dynamics, x1, ū)

# ## objective 

# ## Define aux functions for stage cost and final cost 
function stage_cost(x, u)
    R = I(num_action)
    h = 0.1
    return 0.5 * h * transpose(u) * R * u
end

function final_cost(x)
    Q = Diagonal([100, 10, 1])
    r = x .- [pi/2; 0; 0]
    return 0.5 * (transpose(r) * Q * r)
end

objective = [
    [Cost((x, u) -> stage_cost(x,u), num_state, num_action) for t = 1:T-1]...,
    Cost((x, u) -> final_cost(x), num_state, 0)
]

# intermediate constraint functions 
l = 1
function rx2(x)
    return l * sin(x[1]) + l * sin(x[1] + x[2])
end
function ry2(x)
    return l * cos(x[1]) + l * cos(x[1] + x[2])
end
function rx3(x)
    return rx2(x) + l * sin(x[1]+x[2]+x[3])
end 
function ry3(x)
    return ry2(x) + l * cos(x[1]+x[2]+x[3])
end

# ## constraints
constraints = [
    [Constraint((x, u) -> [
            [u .- 0.10]...;
            [-u .- 0.10]...;
            ry2(x) - 1;
            -ry2(x) - 1;
            ry3(x) - 1;
            -ry3(x) - 1
        ], 
        num_state, num_action, indices_inequality=collect(1:10)) for t = 1:T-1]..., 
        Constraint() # no terminal constraint 
]

# ## solver
solver = Solver(dynamics, objective, constraints)
initialize_controls!(solver, ū) 
initialize_states!(solver, x̄)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

# ## visualize
plot(hcat(x_sol...)')
plot(hcat(u_sol[1:end-1]...)', linetype=:steppost)

# ## benchmark allocations + timing
# using BenchmarkTools
# info = @benchmark solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū))
