using Symbolics
"""
    Double pendulum
"""

struct DoublePendulum{T}
    nq::Int 
    nu::Int 
    nc::Int

    m1::T    # mass link 1
    J1::T    # inertia link 1
    l1::T    # length link 1
    lc1::T   # length to COM link 1

    m2::T    # mass link 2
    J2::T    # inertia link 2
    l2::T    # length link 2
    lc2::T   # length to COM link 2

    g::T     # gravity

    b1::T    # joint friction
    b2::T
end

acrobot_impact = DoublePendulum{Float64}(2, 1, 2,
    1.0, 0.333, 1.0, 0.5, 1.0, 0.333, 1.0, 0.5, 9.81, 0.0, 0.0)
acrobot_normal = DoublePendulum{Float64}(2, 1, 2,
    1.0, 0.333, 1.0, 0.5, 1.0, 0.333, 1.0, 0.5, 9.81, 0.0, 0.0)

function kinematics(model::DoublePendulum, x)
    [model.l1 * sin(x[1]) + model.l2 * sin(x[1] + x[2]),
     -1.0 * model.l1 * cos(x[1]) - model.l2 * cos(x[1] + x[2])]
end

function kinematics_elbow(model::DoublePendulum, x)
    [model.l1 * sin(x[1]),
     -1.0 * model.l1 * cos(x[1])]
end

function M_func(model::DoublePendulum, x)
    a = (model.J1 + model.J2 + model.m2 * model.l1 * model.l1
         + 2.0 * model.m2 * model.l1 * model.lc2 * cos(x[2]))

    b = model.J2 + model.m2 * model.l1 * model.lc2 * cos(x[2])

    c = model.J2

    [a b;
     b c]
end

function τ_func(model::DoublePendulum, x)
    a = (-1.0 * model.m1 * model.g * model.lc1 * sin(x[1])
         - model.m2 * model.g * (model.l1 * sin(x[1])
         + model.lc2 * sin(x[1] + x[2])))

    b = -1.0 * model.m2 * model.g * model.lc2 * sin(x[1] + x[2])

    [a, b]
end

function c_func(model::DoublePendulum, q, q̇)
    a = -2.0 * model.m2 * model.l1 * model.lc2 * sin(q[2]) * q̇[2]
    b = -1.0 * model.m2 * model.l1 * model.lc2 * sin(q[2]) * q̇[2]
    c = model.m2 * model.l1 * model.lc2 * sin(q[2]) * q̇[1]
    d = 0.0

    [a b;
     c d]
end

function B_func(model::DoublePendulum, x)
    [0.0; 1.0]
end

function C_func(model::DoublePendulum, q, q̇)
    c_func(model, q, q̇) * q̇ - τ_func(model, q)
end

function ϕ_func(model, q)
    [0.5 * π - q[2], q[2] + 0.5 * π]
end

function P_func(model, q)
    ϕ = ϕ_func(model, q)
    Symbolics.jacobian(ϕ, q)
end

function lagrangian_derivatives(mass_matrix, dynamics_bias, q, v)
    D1L = -1.0 * dynamics_bias(q, v)
    D2L = mass_matrix(q) * v
    return D1L, D2L
end

function dynamics_acrobot(model::DoublePendulum{T}, mass_matrix, dynamics_bias, dt, q0, q1, u1, λ1, q2) where T
	# evalutate at midpoint
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / dt
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / dt

	D1L1, D2L1 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(mass_matrix, dynamics_bias, qm2, vm2)

	return (-0.5 * dt * D1L1 - 0.5 * dt * D1L2 + D2L2  - D2L1
		- B_func(model, qm2) * u1 * dt
        - transpose(P_func(model, q2)) * λ1 * dt
        + dt * 0.5 .* vm2) # damping
end

function implicit_contact_dynamics(model::DoublePendulum{T}, x, u, dt, μ=0.0) where T
    nq = model.nq
    nu = model.nu
    nc = model.nc

    q0 = x[1:nq]
    q1 = x[nq .+ (1:nq)]

    u1 = u[nu]
    q2 = u[nu .+ (1:nq)]
    s1 = u[(nu + nq) .+ (1:nc)]
    λ1 = u[(nu + nq + nc) .+ (1:nc)]

    [
        dynamics_acrobot(model, a -> M_func(model, a), (a, b) -> C_func(model, a, b),
        dt, q0, q1, u1, λ1, q2);
        (s1 .- ϕ_func(model, q2));
        λ1 .* s1 .- μ;
    ]
end

