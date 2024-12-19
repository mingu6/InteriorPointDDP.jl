"""
    Double pendulum
"""

struct DoublePendulum{T}
    nq::Int 
    nu::Int 
    nc::Int

    m1::T    # mass link 1
    I1::T    # inertia link 1
    l1::T    # length link 1
    lc1::T   # length to COM link 1

    m2::T    # mass link 2
    I2::T    # inertia link 2
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

function M_func(model::DoublePendulum, q)
    a = (model.I1 + model.I2 + model.m2 * model.l1 * model.l1
         + 2.0 * model.m2 * model.l1 * model.lc2 * cos(q[2]))

    b = model.I2 + model.m2 * model.l1 * model.lc2 * cos(q[2])

    c = model.I2

    [a b;
     b c]
end

function τ_g_func(model::DoublePendulum, x)
    a = (-1.0 * model.m1 * model.g * model.lc1 * sin(x[1])
         - model.m2 * model.g * (model.l1 * sin(x[1])
         + model.lc2 * sin(x[1] + x[2])))

    b = -1.0 * model.m2 * model.g * model.lc2 * sin(x[1] + x[2])

    [a, b]
end

function C̃_func(model::DoublePendulum, q, q̇)
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
    C̃_func(model, q, q̇) * q̇ - τ_g_func(model, q)
end

function ϕ_func(model, q)
    [0.5 * π - q[2], q[2] + 0.5 * π]
end

function P_func(model, q)
    [0.0 -1.0; 0.0 1.0]
end

function manipulator_fd(model::DoublePendulum{T}, h, q⁻, q, q⁺, τ, λ) where T
	# evalutate at midpoint
	qᵐ⁻ = 0.5 * (q⁻ + q)
    qᵐ⁺ = 0.5 * (q + q⁺)
    q̇ᵐ⁻ = (q - q⁻) / h
    q̇ᵐ⁺ = (q⁺ - q) / h

	M̂h = M_func(model, qᵐ⁺) * q̇ᵐ⁺ - M_func(model, qᵐ⁻) * q̇ᵐ⁻
	Ĉ = 0.5 * (C_func(model, qᵐ⁺, q̇ᵐ⁺) + C_func(model, qᵐ⁻, q̇ᵐ⁻))
	B = B_func(model, q⁺)
	N = P_func(model, q⁺)
	
	return M̂h + h * (Ĉ - B .* τ - transpose(N) * λ + 0.5 * q̇ᵐ⁺)
end

function implicit_contact_dynamics(model::DoublePendulum{T}, x, u, h, μ=0.0) where T
    nq = model.nq
    nτ = model.nu
    nc = model.nc

    q⁻ = x[1:nq]
    q = x[nq .+ (1:nq)]
    q⁺ = u[nτ .+ (1:nq)]

    τ = u[nτ]
    λ = u[(nτ + nq) .+ (1:nc)]
    s = u[(nτ + nq + nc) .+ (1:nc)]
    
    [
        manipulator_fd(model, h, q⁻, q, q⁺, τ, λ);
        (s .- ϕ_func(model, q⁺));
        λ .* s .- μ;
    ]
end

