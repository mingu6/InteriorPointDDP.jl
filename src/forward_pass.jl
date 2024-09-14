function forward_pass!(policy::PolicyData{T}, problem::ProblemData{T}, data::SolverData{T},
            options::Options{T}; verbose=false) where T
    data.l = 0  # line search iteration counter
    data.status = 0
    data.step_size = T(1.0)
    Δφ = T(0.0)
    μ = data.μ
    τ = max(options.τ_min, T(1.0) - μ)

    θ_prev = data.primal_1_curr
    φ_prev = data.barrier_obj_curr
    θ = θ_prev

    Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, data.step_size)
    Δφ = Δφ_L + Δφ_Q
    min_step_size = estimate_min_step_size(Δφ_L, data, options)

    while data.step_size >= min_step_size
        α = data.step_size
        try
            rollout!(policy, problem, step_size=α)
        catch e
            # reduces step size if NaN or Inf encountered
            e isa DomainError && (data.step_size *= 0.5, continue)
            rethrow(e)
        end
        constraint!(problem, data.μ; mode=:current)
        
        data.status = check_fraction_boundary(problem, policy, τ)
        data.status != 0 && (data.step_size *= 0.5, continue)

        Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, α)
        Δφ = Δφ_L + Δφ_Q
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = constraint_violation_1norm(problem, mode=:current)
        φ = barrier_objective!(problem, data, policy, mode=:current)
        
        # check acceptability to filter
        data.status = !any(x -> (θ >= x[1] && φ >= x[2]), data.filter) ? 0 : 3
        data.status != 0 && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        # check for sufficient decrease conditions for the barrier objective/constraint violation
        data.switching = (Δφ < 0.0) && 
            ((-Δφ) ^ options.s_φ * α^(1-options.s_φ)  > options.δ * θ_prev ^ options.s_θ)
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * Δφ
        if (θ <= data.min_primal_1) && data.switching
            data.status = data.armijo_passed ? 0 : 4  #  sufficient decrease of barrier objective
        else
            suff = (θ <= (1. - options.γ_θ) * θ_prev) || (φ <= φ_prev - options.γ_φ * θ_prev)
            data.status = suff ? 0 : 5
        end
        data.status != 0 && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        data.barrier_obj_next = φ
        data.primal_1_next = θ
        break
    end
    data.step_size < min_step_size && (data.status = 7)
    data.status != 0 && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function check_fraction_boundary(problem::ProblemData{T}, policy::PolicyData{T}, τ::T) where T
    N = problem.horizon

    u = problem.controls
    ū = problem.nominal_controls
    zl = problem.ineq_duals_lo
    zu = problem.ineq_duals_up
    zl̄ = problem.nominal_ineq_duals_lo
    zū = problem.nominal_ineq_duals_up

    bounds = problem.bounds

    status = 0
    for t = 1:N-1
        bt = bounds[t]
        il = bt.indices_lower
        iu = bt.indices_upper
        tmp = policy.u_tmp[t]
        
        tmp .= ū[t]
        tmp .-= bt.lower
        tmp .*= (1. - τ)
        tmp .-= u[t]
        tmp .+= bt.lower
        if any(c > 0.0 for c in @view(tmp[il]))
            status = 2
            break
        end
        
        tmp .= bt.upper
        tmp .-= ū[t]
        tmp .*= (1. - τ)
        tmp .+= u[t]
        tmp .-= bt.upper
        if any(c > 0.0 for c in @view(tmp[iu]))
            status = 2
            break
        end
        
        tmp .= zl̄[t]
        tmp .*= (1. - τ)
        tmp .-= zl[t]
        if any(c > 0.0 for c in @view(tmp[il]))
            status = 2
            break
        end
        
        tmp .= zū[t]
        tmp .*= (1. - τ)
        tmp .-= zu[t]
        if any(c > 0.0 for c in @view(tmp[iu]))
            status = 2
            break
        end
    end
    return status
end

function estimate_min_step_size(Δφ_L::T, data::SolverData{T}, options::Options{T}) where T
    # compute minimum step size based on linear models of step acceptance conditions
    θ_min = data.min_primal_1
    θ = data.primal_1_curr
    γ_θ = options.γ_θ
    γ_α = options.γ_α
    γ_φ = options.γ_φ
    s_θ = options.s_θ
    s_φ = options.s_φ
    δ = options.δ
    if Δφ_L < 0.0 && θ <= θ_min
        min_step_size = min(γ_θ, -γ_φ * θ / Δφ_L, δ * θ ^ s_θ / (-Δφ_L) ^ s_φ)
    elseif Δφ_L < 0.0 && θ > θ_min
        min_step_size = min(γ_θ, -γ_φ * θ / Δφ_L)
    else
        min_step_size = γ_θ
    end
    min_step_size *= γ_α
    min_step_size = max(min_step_size, eps(T))
    return min_step_size
end

function expected_decrease_cost(policy::PolicyData{T}, problem::ProblemData{T}, step_size::T) where T
    Δφ_L = T(0.0)
    Δφ_Q = T(0.0)
    N = problem.horizon
    Qu = policy.hamiltonian.gradient_control
    Quu = policy.hamiltonian.hessian_control_control
    gains = policy.gains_data
    
    for t = N-1:-1:1
        Δφ_L += dot(Qu[t], gains.α[t])
        Δφ_Q += 0.5 * dot(gains.α[t], Quu[t], gains.α[t])
    end
    return Δφ_L * step_size, Δφ_Q * step_size^2
end

function rollout!(policy::PolicyData{T}, problem::ProblemData{T}; step_size::T=1.0) where T
    dynamics = problem.model.dynamics
    
    x, u, _ = primal_trajectories(problem, mode=:current)
    ϕ, zl, zu = dual_trajectories(problem, mode=:current)
    x̄, ū, _ = primal_trajectories(problem, mode=:nominal)
    ϕ̄, zl̄, zū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    α, ψ = policy.gains_data.α, policy.gains_data.ψ
    β, ω = policy.gains_data.β, policy.gains_data.ω
    χl, χu = policy.gains_data.χl, policy.gains_data.χu
    ζl, ζu = policy.gains_data.ζl, policy.gains_data.ζu

    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + β[t] * (x[t] - x̄[t]) + step_size * α[t]
        u[t] .= α[t]
        u[t] .*= step_size
        u[t] .+= ū[t]
        mul!(u[t], β[t], x[t], 1.0, 1.0)
        mul!(u[t], β[t], x̄[t], -1.0, 1.0)

        # ϕ[t] .= ϕ̄[t] + ω[t] * (x[t] - x̄[t]) + step_size * ψ[t]
        ϕ[t] .= ψ[t]
        ϕ[t] .*= step_size
        ϕ[t] .+= ϕ̄[t]
        mul!(ϕ[t], ω[t], x[t], 1.0, 1.0)
        mul!(ϕ[t], ω[t], x̄[t], -1.0, 1.0)

        zl[t] .= χl[t]
        zl[t] .*= step_size
        zl[t] .+= zl̄[t]
        mul!(zl[t], ζl[t], x[t], 1.0, 1.0)
        mul!(zl[t], ζl[t], x̄[t], -1.0, 1.0)

        zu[t] .= χu[t]
        zu[t] .*= step_size
        zu[t] .+= zū[t]
        mul!(zu[t], ζu[t], x[t], 1.0, 1.0)
        mul!(zu[t], ζu[t], x̄[t], -1.0, 1.0)
        
        dynamics!(d, x[t+1], x[t], u[t])
    end
end
