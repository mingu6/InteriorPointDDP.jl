function setup_acrobot(; horizon::Int=51, dt::Float64=0.1, x1::Vector{Float64}=zeros(Float64, 4))
    N = horizon
    num_state = 4
    num_action = 1
    xN = [0.0; π; 0.0; 0.0]
    
    function acrobot_continuous(x, u)
        mass1 = 1.0  
        inertia1 = 0.33  
        length1 = 1.0 
        lengthcom1 = 0.5 
    
        mass2 = 1.0  
        inertia2 = 0.33  
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
    
    acrobot_discrete = (x, u) -> x + dt * acrobot_continuous(x + 0.5 * dt * acrobot_continuous(x, u), u)
    acrobot_dyn = Dynamics(acrobot_discrete, num_state, num_action)
    dynamics = [acrobot_dyn for k = 1:N-1]
    
    stage = Cost((x, u) -> 0.01 * dot(x[3:4], x[3:4]) + 0.01 * dot(u, u), num_state, num_action)
    objective = [
        [stage for k = 1:N-1]...,
        Cost((x, u) -> 10.0 * dot(x - xN, x - xN), num_state, 0),
    ] 
    
    stage_constr = Constraint((x, u) -> [
        u[1] - 5.0; 
        -u[1] - 5.0
    ], 
    num_state, num_action, indices_inequality=collect(1:2))
    
    constraints = [
        [stage_constr for k = 1:N-1]..., 
            Constraint() # no terminal constraint 
    ]
    
    return BenchmarkProblem(N, dt, num_state, num_action, x1, dynamics, objective, constraints)
end
