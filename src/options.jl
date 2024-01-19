Base.@kwdef mutable struct Options{T}
    feasible::Bool = false  # feasible start IPDDP
    optimality_tolerance::T = 1.0e-7
    max_iterations::Int = 1000
    min_step_size::T = 1.0e-8
    reset_cache::Bool = true
    verbose = true
    # Regularisation Params
    reg::Int8 = 0
    start_reg::Int8 = 0
    end_reg::Int8 = 24
    reg_step::Int8 = 1
    # dual/slack initialization
    κ_1::Float64 = 0.1  # dual (s)
    κ_2::Float64 = 0.1  # slack (y)
    # optimality condition parameters
    κ_ϵ::Float64 = 10
    κ_μ::Float64 = 0.2
    θ_μ::Float64 = 1.5
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