Base.@kwdef mutable struct Options{T}
    feasible::Bool = true  # feasible start IPDDP
    gauss_newton::Bool = true # disregard 2nd order dynamics
    optimality_tolerance::T = 1.0e-7
    max_iterations::Int = 500
    reset_cache::Bool = true
    verbose = true
    # Regularisation schedule
    ϕ_1::Float64 = 1e-4
    ϕ_min::Float64 = 1e-20
    ϕ_max::Float64 = 1e40
    ψ_p_1::Float64 = 100
    ψ_p_2::Float64 = 8
    ψ_m::Float64 = 0.3333333
    # dual/slack initialization
    κ_1::Float64 = 0.1  # dual (s)
    κ_2::Float64 = 0.01  # slack (y)
    # optimality condition parameters
    κ_ϵ::Float64 = 0.2
    κ_μ::Float64 = 0.2
    θ_μ::Float64 = 1.2
    s_max::Float64 = 100
    # fraction to boundary condition
    τ_min::Float64 = 0.99
    # initial perturbation
    μ_0::Float64 = 0.1  # multiplier on perturbation initialisation
    # sufficient conditions for step acceptance
    η_φ::Float64 = 1e-4  # armijo
    s_φ::Float64 = 2.3
    δ::Float64 = 1.0
    s_θ::Float64 = 1.1
    γ_α::Float64 = 0.05  # for min. step size calibration
    γ_θ::Float64 = 1e-5
    γ_φ::Float64 = 1e-5
    # misc.
    κ_Σ::Float64 = 1e10  # dual variable rescaling threshold param.
end 