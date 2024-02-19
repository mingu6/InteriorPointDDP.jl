function setup_cartpole(; horizon::Int=51, dt::Float64=0.1, x1::Vector{Float64}=zeros(Float64, 4))
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
    
    N = horizon
    num_state = 4 
    num_action = 1
    xN = [0.0; π; 0.0; 0.0]
    
    cartpole_discrete = (x, u) -> x + dt * cartpole_continuous(x + 0.5 * dt * cartpole_continuous(x, u), u)
    cartpole_dyn = Dynamics(cartpole_discrete, num_state, num_action)
    dynamics = [cartpole_dyn for k = 1:N-1]

    stage = Cost((x, u) -> 0.01 * dot(x - xN, x - xN) + 0.1 * dot(u, u), num_state, num_action)
    objective = [
        [stage for k = 1:N-1]...,
        Cost((x, u) -> 100.0 * dot(x - xN, x - xN), num_state, 0),
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
