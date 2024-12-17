function forward_pass!(update_rule::UpdateRuleData{T}, problem::ProblemData{T}, data::SolverData{T},
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
    
    Δφ = expected_decrease_cost(update_rule, problem, data.step_size)
    min_step_size = estimate_min_step_size(Δφ, data, options)

    while data.step_size >= min_step_size
        α = data.step_size
        try
            rollout!(update_rule, data, problem, step_size=α)
        catch e
            # reduces step size if NaN or Inf encountered
            e isa DomainError && (data.step_size *= 0.5, continue)
            rethrow(e)
        end
        constraint!(problem, data.μ; mode=:current)
        
        data.status = check_fraction_boundary(problem, update_rule, τ)
        data.status != 0 && (data.step_size *= 0.5, continue)

        Δφ = expected_decrease_cost(update_rule, problem, α)
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = constraint_violation_1norm(problem, mode=:current)
        φ = barrier_objective!(problem, data, update_rule, mode=:current)
        
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

function check_fraction_boundary(problem::ProblemData{T}, update_rule::UpdateRuleData{T}, τ::T) where T
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
        tmp = update_rule.u_tmp[t]
        
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

function expected_decrease_cost(update_rule::UpdateRuleData{T}, problem::ProblemData{T}, step_size::T) where T
    Δφ = T(0.0)
    N = problem.horizon
    Q̂u = update_rule.hamiltonian.gradient_control
    parameters = update_rule.parameters
    
    for t = N-1:-1:1
        Δφ += dot(Q̂u[t], parameters.α[t])
    end
    return Δφ * step_size
end

function rollout!(update_rule::UpdateRuleData{T}, data::SolverData{T}, problem::ProblemData{T}; step_size::T=1.0) where T
    dynamics = problem.model.dynamics
    
    x, u, _ = primal_trajectories(problem, mode=:current)
    ϕ, zl, zu = dual_trajectories(problem, mode=:current)
    x̄, ū, _ = primal_trajectories(problem, mode=:nominal)
    ϕ̄, zl̄, zū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    α, ψ = update_rule.parameters.α, update_rule.parameters.ψ
    β, ω = update_rule.parameters.β, update_rule.parameters.ω
    χl, χu = update_rule.parameters.χl, update_rule.parameters.χu
    ζl, ζu = update_rule.parameters.ζl, update_rule.parameters.ζu

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
        
        fn_eval_time_ = time()
        dynamics!(d, x[t+1], x[t], u[t])
        data.fn_eval_time += time() - fn_eval_time_
    end
end
