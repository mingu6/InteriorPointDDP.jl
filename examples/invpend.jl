function setup_invpend(; horizon::Int=501, dt::Float64=0.05, x1::Vector{Float64}=[-pi; 0.0])
    N = horizon
    num_state = 2
    num_action = 1
    
    function invpend_dynamics(x, u)
        return [
            x[1] .+ dt * x[2];
            x[2] .+ dt * sin(x[1]) .+ dt * u;
        ]
    end
    
    invpend = Dynamics(invpend_dynamics, num_state, num_action)
    dynamics = [invpend for k = 1:N-1] 
    
    function stage_cost(x, u)
        Q = I(num_state)
        R = I(num_action)
        return 0.5 * dt * (transpose(x) * Q * x + transpose(u) * R * u)
    end
    
    function final_cost(x)
        P = 10 * I(num_state)
        return 0.5 * transpose(x) * P * x
    end
    
    stage = Cost((x, u) -> stage_cost(x,u), num_state, num_action)
    objective = [
        [stage for k = 1:N-1]...,
        Cost((x, u) -> final_cost(x), num_state, 0)
    ]
    
    # ## constraints
    stage_constr = Constraint((x, u) -> [
        u[1] - 0.25; 
        -u[1] - 0.25;
    ], 
    num_state, num_action, indices_inequality=collect(1:2))
    
    constraints = [
        [stage_constr for k = 1:N-1]..., 
            Constraint() # no terminal constraint 
    ]
    return BenchmarkProblem(N, dt, num_state, num_action, x1, dynamics, objective, constraints)
end
