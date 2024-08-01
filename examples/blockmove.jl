function setup_blockmove(; horizon::Int=25, dt::Float64=0.04, x1::Vector{Float64}=zeros(Float64, 2))
    function blockmove_continuous(x, u)
        return [x[2], u[1]]
    end

    blockmove_discrete = (x, u) -> x + dt * blockmove_continuous(x + 0.5 * dt * blockmove_continuous(x, u), u)

    blockmove_dyn = Dynamics(blockmove_discrete, 2, 3)
    dynamics = [blockmove_dyn for k = 1:horizon-1]

    xN = [1.0; 0.0]

    stage = Cost((x, u) -> u[2] + u[3], 2, 3)
    objective = [
        [stage for k = 1:horizon-1]...,
        Cost((x, u) -> 0.0, 2, 0),
    ]

    stage_constr = Constraint((x, u) -> [
        u[2] - u[3] - u[1] * x[2]
    ], 
    2, 3, 
    bounds_lower=[-10.0, 0.0, 0.0], bounds_upper=[10.0, Inf, Inf])

    term_constr = Constraint((x, u) -> [
        dot(blockmove_discrete(x, u), xN) - dot(xN, xN),
        u[2] - u[3] - u[1] * x[2]
    ], 
    2, 3, 
    bounds_lower=[-10.0, 0.0, 0.0], bounds_upper=[10.0, Inf, Inf])

    constraints = [
    [stage_constr for k = 1:horizon-2]..., 
        term_constr
    ]
    
   return BenchmarkProblem(horizon, dt, 2, 3, x1, dynamics, objective, constraints)
end