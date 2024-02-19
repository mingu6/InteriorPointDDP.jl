function setup_car(; horizon::Int=501, dt::Float64=0.1, x1::Vector{Float64}=[1.0; 1.0; pi*3/2; 0.0])
    N = horizon
    num_state = 4
    num_action = 2
    h = dt
    
    function car_aux(x, u)
        d = 2.0  
    
        return [
            (d + h*x[4]*cos(u[1]) - sqrt(d^2 - (h*x[4]*sin(u[1]))^2))*cos(x[3]);
            (d + h*x[4]*cos(u[1]) - sqrt(d^2 - (h*x[4]*sin(u[1]))^2))*sin(x[3]);
            asin(sin(u[1])*h*x[4]/d);
            h*u[2]
        ]
    end
    
    function car_dynamics(x, u)
        x + car_aux(x, u)
    end
    
    function stage_cost(x, u)
        R = 1.0e-2 * Diagonal([1, 0.01])
        Q = 1.0e-3 * [1; 1]
    
        c1 = u' * R * u
        c2 = 1.0e-3 * (sqrt(x[1]^2+Q[1])-Q[1]) + 1.0e-3 * (sqrt(x[2]^2+Q[2])-Q[2])
        return c1 + c2
    end
    
    function final_cost(x)
        P = [0.01; 0.01; 0.01; 1]
        p = 0.1 * (sqrt(x[1]^2+P[1])-P[1])+
            0.1 * (sqrt(x[2]^2+P[2])-P[2])+
            1 * (sqrt(x[3]^2+P[3])-P[3])+
            0.3 * (sqrt(x[4]^2+P[4])-P[4])
        return p
    end
    
    car = Dynamics(car_dynamics, num_state, num_action)
    dynamics = [car for k = 1:N-1] 
    
    objective = [
        [Cost((x, u) -> stage_cost(x,u), num_state, num_action) for k = 1:N-1]...,
        Cost((x, u) -> final_cost(x), num_state, 0)
    ]
    
    # ## constraints
    constraints = [
        [Constraint((x, u) -> [
                u[1] - 0.5; 
                -u[1] - 0.5; 
                u[2] - 2; 
                -u[2] - 2
            ], 
            num_state, num_action, indices_inequality=collect(1:4)) for k = 1:N-1]..., 
            Constraint() # no terminal constraint 
    ]
    return BenchmarkProblem(N, dt, num_state, num_action, x1, dynamics, objective, constraints)
end
