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
    
    Δφ = expected_decrease_objective(update_rule, problem)

    while data.step_size >= eps(T)
        γ = data.step_size
        try
            rollout!(update_rule, data, problem, step_size=γ)
        catch e
            # reduces step size if NaN or Inf encountered
            e isa DomainError && (data.step_size *= 0.5, continue)
            rethrow(e)
        end
        
        data.status = check_fraction_boundary(problem, update_rule, τ)
        data.status != 0 && (data.step_size *= 0.5, continue)

        constraint!(problem, data.μ; mode=:current)
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = constraint_violation_1norm(problem, mode=:current)
        φ = barrier_objective!(problem, data, update_rule, mode=:current)
        
        # check acceptability to filter
        data.status = !any(x -> (θ >= x[1] && φ >= x[2]), data.filter) ? 0 : 3
        data.status != 0 && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        # check for sufficient decrease conditions for the barrier objective/constraint violation
        data.switching = (Δφ < 0.0) && 
            ((-γ * Δφ) ^ options.s_φ * γ^(1-options.s_φ)  > options.δ * θ_prev ^ options.s_θ)
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * γ * Δφ
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
    data.step_size < eps(T) && (data.status = 7)
    data.status != 0 && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function check_fraction_boundary(problem::ProblemData{T}, update_rule::UpdateRuleData{T}, τ::T) where T
    N = problem.horizon

    _, _, _, il, iu = primal_trajectories(problem, mode=:current)
    _, zl, zu = dual_trajectories(problem, mode=:current)
    _, _, _, il̄, iū = primal_trajectories(problem, mode=:nominal)
    _, zl̄, zū = dual_trajectories(problem, mode=:nominal)

    for t = 1:N
        if any(c * (1. - τ) > d for (c, d) in zip(il̄[t], il[t]))
            return 2
        end

        if any(c * (1. - τ) > d for (c, d) in zip(iū[t], iu[t]))
            return 2
        end

        if any(c * (1. - τ) > d for (c, d) in zip(zl̄[t], zl[t]))
            return 2
        end

        if any(c * (1. - τ) > d for (c, d) in zip(zū[t], zu[t]))
            return 2
        end
    end
    return 0
end

function expected_decrease_objective(update_rule::UpdateRuleData{T}, problem::ProblemData{T}) where T
    Δφ = T(0.0)
    N = problem.horizon
    
    for t = N:-1:1
        Δφ += dot(update_rule.ĝ[t], update_rule.parameters.α[t])
        # Δφ += dot(problem.nominal_constr[t], update_rule.parameters.ψ[t])
    end
    return Δφ
end

function rollout!(update_rule::UpdateRuleData{T}, data::SolverData{T}, problem::ProblemData{T}; step_size::T=1.0) where T
    dynamics = problem.model.dynamics
    bounds = problem.bounds
    N = problem.horizon
    
    x, u, _, il, iu = primal_trajectories(problem, mode=:current)
    ϕ, zl, zu = dual_trajectories(problem, mode=:current)
    x̄, ū, _, _, _ = primal_trajectories(problem, mode=:nominal)
    ϕ̄, zl̄, zū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    α, ψ = update_rule.parameters.α, update_rule.parameters.ψ
    β, ω = update_rule.parameters.β, update_rule.parameters.ω
    χl, χu = update_rule.parameters.χl, update_rule.parameters.χu
    ζl, ζu = update_rule.parameters.ζl, update_rule.parameters.ζu
    
    δx = update_rule.x_tmp

    for t in 1:N
        δx[t] .= x[t]
        δx[t] .-= x̄[t]

        # u[t] .= ū[t] + β[t] * (x[t] - x̄[t]) + step_size * α[t]
        u[t] .= α[t]
        u[t] .*= step_size
        u[t] .+= ū[t]
        mul!(u[t], β[t], δx[t], 1.0, 1.0)

        # ϕ[t] .= ϕ̄[t] + ω[t] * (x[t] - x̄[t]) + step_size * ψ[t]
        ϕ[t] .= ψ[t]
        ϕ[t] .*= step_size
        ϕ[t] .+= ϕ̄[t]
        mul!(ϕ[t], ω[t], δx[t], 1.0, 1.0)

        zl[t] .= χl[t]
        zl[t] .*= step_size
        zl[t] .+= zl̄[t]
        mul!(zl[t], ζl[t], δx[t], 1.0, 1.0)

        zu[t] .= χu[t]
        zu[t] .*= step_size
        zu[t] .+= zū[t]
        mul!(zu[t], ζu[t], δx[t], 1.0, 1.0)
        
        fn_eval_time_ = time()
        t < N && dynamics!(dynamics[t], x[t+1], x[t], u[t])
        
        # evaluate inequality constraints
        il[t] .= u[t]
        il[t] .-= bounds[t].lower
        iu[t] .= bounds[t].upper
        iu[t] .-= u[t]
        data.fn_eval_time += time() - fn_eval_time_
    end
end
