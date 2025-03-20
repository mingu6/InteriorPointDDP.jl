struct Cartpole{T}
    # dimensions
    nq::Int # generalized coordinates
    nu::Int # controls

    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
end

function kinematics(model::Cartpole, q)
    [q[1] + model.l * sin(q[2]); -model.l * cos(q[2])]
end

function M_func(model::Cartpole, q)
    H = [model.mc + model.mp model.mp * model.l * cos(q[2]);
		 model.mp * model.l * cos(q[2]) model.mp * model.l^2.0]
    return H
end

function B_func(model::Cartpole, q)
    [1.0; 0.0]
end

function C_func(model::Cartpole, q, q̇)
    C = [0.0 -1.0 * model.mp * q̇[2] * model.l * sin(q[2]);
	 	 0.0 0.0]
    G = [0.0,
		 -model.mp * model.g * model.l * sin(q[2])]
    return C * q̇ - G
end

function implicit_dynamics(model::Cartpole{T}, x, y) where T
    nq = model.nq
    nu = model.nu
    q = x[1:nq]
    q̇ = x[nq .+ (1:nq)]
    q̈ = y[nu .+ (1:nq)]
    u = y[1:nu]
    M = M_func(model, q)
    B = B_func(model, q)
    dyn_bias = C_func(model, q, q̇)
    return M * q̈ + dyn_bias - B .* u
end

function forward_dynamics(model::Cartpole{T}, x, u) where T
    nq = model.nq
    q = x[1:nq]
    q̇ = x[nq .+ (1:nq)]
    M = M_func(model, q)
    B = B_func(model, q)
    dyn_bias = C_func(model, q, q̇)
    return M \ (B .* u - dyn_bias)
end
