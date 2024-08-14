function setup_cartpole(; horizon::Int=101, dt::Float64=0.05, x1::Vector{Float64}=zeros(Float64, 4))
    function cartpole_continuous(x, u)
        mc = 1.0  # mass of the cart in kg (10)
        mp = 0.3   # mass of the pole (point mass at the end) in kg
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
    num_action = 2 # one is linear force, one is slack for state ineq. constraint
    xN = [0.0; π; 0.0; 0.0]
    
    # cartpole_discrete = (x, u) -> x + dt * cartpole_continuous(x + 0.5 * dt * cartpole_continuous(x, u), u)
    cartpole_discrete = (x, u) -> x + dt * cartpole_continuous(x, u)
    cartpole_dyn = Dynamics(cartpole_discrete, num_state, num_action)
    dynamics = [[cartpole_dyn for k = 1:N-2]..., Dynamics(cartpole_discrete, num_state, num_action-1)]

    stage = Cost((x, u) -> dt * dot(u[1], u[1]), num_state, num_action)
    objective = [
        [stage for k = 1:N-2]..., Cost((x, u) -> dt * (dot(u[1], u[1])), num_state, num_action-1),
        # Cost((x, u) -> 0.0, num_state, 0),
        Cost((x, u) -> 400. * dot(x - xN, x - xN), num_state, 0),
    ] 
    
    stage_constr = Constraint((x, u) -> [
        cartpole_discrete(x, u)[1] - u[2]
    ], 
    num_state, num_action, 
    # bounds_lower=[-3., -0.15], bounds_upper=[3., 0.1])
    bounds_lower=[-3., -0.10], bounds_upper=[3., 0.2])

    # term_constr = Constraint((x, u) -> [dot(cartpole_discrete(x, u) - xN, cartpole_discrete(x, u) - xN)],
    #     num_state, num_action - 1, 
    #     bounds_lower=[-20.], bounds_upper=[20.]
    # )

    term_constr = Constraint((x, u) -> [],
        num_state, num_action - 1, 
        bounds_lower=[-3.], bounds_upper=[3.]
    )
    
    constraints = [
    [stage_constr for k = 1:N-2]..., 
        term_constr
    ]
    
   return BenchmarkProblem(N, dt, num_state, num_action, x1, dynamics, objective, constraints)
end
