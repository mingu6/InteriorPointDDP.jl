Base.@kwdef mutable struct Options{T}
    feasible::Bool = true
    quasi_newton::Bool = true
    optimality_tolerance::T = 1.0e-7
    max_iterations::Int = 1000
    reset_cache::Bool = true
    verbose = false
    print_frequency = 10
    
    μ_init::Float64 = 0.1                # multiplier on barrier parameter initialisation
    ineq_dual_init::Float64 = 0.1        # dual variable initialisation value TODO: change name
    slack_init::Float64 = 0.01           # slack variable initialisation value
    
    reg_state::Bool = false              # apply regularisation to V̂xx instead of Q̂uu (Tassa)
    ϕ_1::Float64 = 1e-4
    ϕ_min::Float64 = 1e-20
    ϕ_max::Float64 = 1e40
    ψ_p_1::Float64 = 100.0
    ψ_p_2::Float64 = 8.0
    ψ_m::Float64 = 1.0 / 3.0

    κ_ϵ::Float64 = 0.2                   # tolerance factor for lowering barrier parameter
    κ_μ::Float64 = 0.2                   # linear decrease factor for barrier parameter
    θ_μ::Float64 = 1.2                   # superlinear decrease factor for barrier parameter
    τ_min::Float64 = 0.99                # lower bound on fraction-to-boundary parameter
    
    s_max::Float64 = 100.0               # scaling threshold for NLP error
    η_φ::Float64 = 1e-8                  # relaxation factor in the armijo condition
    s_φ::Float64 = 2.3                   # exponent for linear barrier function model in switching rule
    δ::Float64 = 1.0                     # multiplier for constraint violation in the switching rule
    s_θ::Float64 = 1.1                   # exponent for current constraint violation in the switching rule
    γ_α::Float64 = 0.05                  # safety factor for minimum step size (in (0, 1))
    γ_θ::Float64 = 1e-5                  # relaxation factor in the filter margin for constraint violation (in (0, 1))
    γ_φ::Float64 = 1e-5                  # relaxation factor in the filter margin for barrier function (in (0, 1))

    κ_Σ::Float64 = 1e10                  # dual variable rescaling threshold param.
end 