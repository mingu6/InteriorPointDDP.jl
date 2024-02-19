function setup_arm(; horizon::Int=501, dt::Float64=0.1, x1::Vector{Float64}=[-pi/2; 0.0; 0.0])
    N = horizon
    num_state = 3 
    num_action = 3 
    
    function arm_dynamics(x, u)
        dt = 0.1
        return x .+ dt .* u
    end
    
    arm = Dynamics(arm_dynamics, num_state, num_action) 
    dynamics = [arm for k = 1:N-1] 
    
    function stage_cost(x, u)
        R = I(num_action)
        return 0.5 * dt * transpose(u) * R * u
    end
    
    function final_cost(x)
        Q = Diagonal([100, 10, 1])
        r = x .- [pi/2; 0; 0]
        return 0.5 * (transpose(r) * Q * r)
    end
    
    objective = [
        [Cost((x, u) -> stage_cost(x,u), num_state, num_action) for k = 1:N-1]...,
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
            num_state, num_action, indices_inequality=collect(1:10)) for k = 1:N-1]..., 
            Constraint() # no terminal constraint 
    ]
    return BenchmarkProblem(N, dt, num_state, num_action, x1, dynamics, objective, constraints)
end
