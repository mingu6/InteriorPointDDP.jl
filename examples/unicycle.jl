function setup_unicycle(; horizon::Int=401, dt::Float64=0.02, x1::Vector{Float64}=[-10.0; 0.0; 0.0])
    N = horizon
    num_state = 3
    num_action = 1
    
    function unicycle_discrete(x, u)
        v = 1.5
        φ = x[3]
        return [x[1] + dt * v * cos(φ), x[2] + dt * v * sin(φ), φ + dt * u[1]]
    end
    
    unicycle_dyn = Dynamics(unicycle_discrete, num_state, num_action)
    dynamics = [unicycle_dyn for k = 1:N-1]

    stage = Cost((x, u) -> 0.001 * dot(x, x) + 0.0001 * dot(u, u), num_state, num_action)
    objective = [
        [stage for k = 1:N-1]...,
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
        [stage_constr for k = 1:N-1]..., 
            Constraint()
    ]
    return BenchmarkProblem(N, dt, num_state, num_action, x1, dynamics, objective, constraints)
end
