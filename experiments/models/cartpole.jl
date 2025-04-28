struct Cartpole{T}
    # dimensions
    nq::Int # generalized coordinates
    nu::Int # controls
    nc::Int # contact points

    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2

    friction::Vector{T} # friction coefficients for slider and arm joints
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

function P_func(model::Cartpole, x)
    [1.0 0.0;
     0.0 1.0]
end

function C_func(model::Cartpole, q, q̇)
    C = [0.0 -1.0 * model.mp * q̇[2] * model.l * sin(q[2]);
	 	 0.0 0.0]
    G = [0.0,
		 -model.mp * model.g * model.l * sin(q[2])]
    return C * q̇ - G
end

function manipulator_fd(model::Cartpole{T}, Δ , q⁻, q, q⁺, F, λ) where T
	# evalutate at midpoint
	qᵐ⁻ = 0.5 * (q⁻ + q)
    qᵐ⁺ = 0.5 * (q + q⁺)
    q̇ᵐ⁻ = (q - q⁻) / Δ
    q̇ᵐ⁺ = (q⁺ - q) / Δ

	M̂Δ = M_func(model, qᵐ⁺) * q̇ᵐ⁺ - M_func(model, qᵐ⁻) * q̇ᵐ⁻
	Ĉ = 0.5 * (C_func(model, qᵐ⁺, q̇ᵐ⁺) + C_func(model, qᵐ⁻, q̇ᵐ⁻))
	B = B_func(model, q⁺)
	N = P_func(model, q⁺)

	return M̂Δ + Δ  * (Ĉ - B .* F - transpose(N) * λ)
end

function implicit_contact_dynamics(model::Cartpole{T}, x, u, Δ, μ=0.0) where T
    nq = model.nq
    nF = model.nu
    nc = model.nc

    q⁻ = x[1:nq]
    q = x[nq .+ (1:nq)]
    q⁺ = u[nF .+ (1:nq)]

    q̇ᵐ⁺ = (q⁺ - q) / Δ

    F = u[nF]
    β1 = u[(nF + nq) .+ (1:nc)]
    β2 = u[(nF + nq + nc) .+ (1:nc)]
    η1 = u[(nF + nq + 2 * nc) .+ (1:nc)]
    η2 = u[(nF + nq + 3 * nc) .+ (1:nc)]
    ψ = u[(nF + nq + 4 * nc) .+ (1:nc)]
    s = u[(nF + nq + 5 * nc) .+ (1:nc)]

    λ = [β1[1] - β1[2]; β2[1] - β2[2]]

    γ1 = model.friction[1] * (model.mp + model.mc) * model.g * Δ
    γ2 = model.friction[2] * model.mp * model.g * model.l * Δ

    [
        manipulator_fd(model, Δ , q⁻, q, q⁺, F, λ);
        [q̇ᵐ⁺[1]; -q̇ᵐ⁺[1]] .+ ψ[1] - η1;
        [q̇ᵐ⁺[2]; -q̇ᵐ⁺[2]] .+ ψ[2] - η2;
        γ1 - sum(β1) - s[1];
        γ2 - sum(β2) - s[2];
        ψ[1] * s[1] - μ;
        ψ[2] * s[2] - μ;
        β1 .* η1 .- μ;
        β2 .* η2 .- μ
    ]
end

function implicit_contact_dynamics_slack(model::Cartpole{T}, x, u, Δ) where T
    nq = model.nq
    nF = model.nu
    nc = model.nc

    q⁻ = x[1:nq]
    q = x[nq .+ (1:nq)]
    q⁺ = u[nF .+ (1:nq)]

    q̇ᵐ⁺ = (q⁺ - q) / Δ

    F = u[nF]
    β1 = u[(nF + nq) .+ (1:nc)]
    β2 = u[(nF + nq + nc) .+ (1:nc)]
    η1 = u[(nF + nq + 2 * nc) .+ (1:nc)]
    η2 = u[(nF + nq + 3 * nc) .+ (1:nc)]
    ψ = u[(nF + nq + 4 * nc) .+ (1:nc)]
    s = u[(nF + nq + 5 * nc) .+ (1:nc)]
    sc = u[(nF + nq + 6 * nc) .+ (1:6)]

    λ = [β1[1] - β1[2]; β2[1] - β2[2]]

    γ1 = model.friction[1] * (model.mp + model.mc) * model.g * Δ
    γ2 = model.friction[2] * model.mp * model.g * model.l * Δ

    [
        manipulator_fd(model, Δ , q⁻, q, q⁺, F, λ);
        [q̇ᵐ⁺[1]; -q̇ᵐ⁺[1]] .+ ψ[1] - η1;
        [q̇ᵐ⁺[2]; -q̇ᵐ⁺[2]] .+ ψ[2] - η2;
        γ1 - sum(β1) - s[1];
        γ2 - sum(β2) - s[2];
        ψ[1] * s[1] - sc[1];
        ψ[2] * s[2] - sc[2];
        β1 .* η1 .- sc[3:4];
        β2 .* η2 .- sc[5:6]
    ]
end
