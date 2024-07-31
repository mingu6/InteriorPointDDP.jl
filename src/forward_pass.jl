function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; verbose=false)
    constr_data = problem.constraints
    
    data.l = 0  # line search iteration counter
    data.status = true
    data.step_size = 1.0
    min_step_size = -Inf
    Δφ = 0.0
    μ = data.μ
    τ = max(options.τ_min, 1.0 - μ)

    if min_step_size == -Inf
        Δφ_L, _ = expected_decrease_cost(policy, problem, data.step_size)
        min_step_size = estimate_min_step_size(Δφ_L, data, options)
    end

    while data.step_size >= min_step_size # check whether we still want it to be this
        α = data.step_size
        try
            rollout!(policy, problem, τ, step_size=α)
        catch
            # reduces step size if NaN or Inf encountered
            data.step_size *= 0.5
            # data.l += 1
            continue
        end
        constraint!(problem, mode=:current)
        
        data.status = check_fraction_boundary(problem, τ)
        # if data.k == 1
        #     println("frac ", data.k, " ", data.l, " ", data.status, " ", τ, " ", data.min_primal_1)
        # end
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)
        # !data.status && (data.step_size *= 0.5, continue)

        Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, α)
        Δφ = Δφ_L + Δφ_Q

        # println("exp ", data.k, " ", data.l, " ", α, " ", Δφ, " ", μ)
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = constraint_violation_1norm(problem, mode=:current)
        φ = barrier_objective!(problem, data, mode=:current)
        θ_prev = data.primal_1_curr
        φ_prev = data.barrier_obj_curr
        
        # check acceptability to filter
        data.status = !any(x -> all([θ, φ] .>= x), data.filter)
        # println("filter ", data.k, " ", data.status, " ", α, " ", θ, " ", φ, " ", data.filter)
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        # check for sufficient decrease conditions for the barrier objective/constraint violation
        data.switching = (Δφ < 0.0) && 
            ((-Δφ) ^ options.s_φ * α^(1-options.s_φ)  > options.δ * θ_prev ^ options.s_θ)
        # println("switch ", data.switching, " ", θ_prev, " ", -Δφ)
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * Δφ
        # println("armijo ", data.armijo_passed, " ", θ <= data.min_primal_1, " ", φ - φ_prev, " ", Δφ)
        if (θ <= data.min_primal_1) && data.switching
            data.status = data.armijo_passed  #  sufficient decrease of barrier objective
        else
            suff = (θ <= (1. - options.γ_θ) * θ_prev) || (φ <= φ_prev - options.γ_φ * θ_prev)
            # println("suff ", θ, " ", (1. - options.γ_θ) * θ_prev, " ", φ, " ", φ_prev - options.γ_φ * θ_prev)
            !suff && (data.status = false)
        end
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        data.barrier_obj_next = φ
        data.primal_1_next = θ
        break
    end
    # println(data.l, " ", data.step_size, " ", min_step_size)
    data.step_size < min_step_size && (data.status = false)
    !data.status && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function check_fraction_boundary(problem::ProblemData, τ::Float64)
    N = problem.horizon
    x, u, _, il, iu = primal_trajectories(problem, mode=:current)
    _, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, _, il̄, iū = primal_trajectories(problem, mode=:nominal)
    _, vl̄, vū = dual_trajectories(problem, mode=:nominal)

    # check_fn = k -> any(il[k] .< (1. - τ) .*  il̄[k]) || any(iu[k] .< (1. - τ) .*  iū[k]) || any(vl[k] .< (1. - τ) .*  vl̄[k]) || any(vu[k] .< (1. - τ) .*  vū[k])
    # println(" ")
    # println(" ")
    status = true
    for k = 1:N-1
        if any((il[k] .< (1. - τ) .*  il̄[k]) .* .!isinf.(il̄[k]))
            status = false
            # println("il: ", k, " ", il[k], " ", il̄[k], " ", u[k], " ", ū[k], " ", x[k], " ", x̄[k])
        elseif any((iu[k] .< (1. - τ) .*  iū[k]) .* .!isinf.(iū[k]))
            # println("iu: ", k, " ", iu[k], " ", iū[k])
            status = false
            # break
        elseif any((vl[k] .< (1. - τ) .*  vl̄[k]) .* .!isinf.(vl̄[k]))
            # println("vl: ", k, " ", vl[k], " ", vl̄[k])
            status = false
            # break
        elseif any((vu[k] .< (1. - τ) .*  vū[k]) .* .!isinf.(vū[k]))
            # println("vu: ", k, " ", vu[k], " ", vū[k])
            status = false
            # break
        end
    end
    return status
    return !any(check_fn, 1:N-1)
end

function estimate_min_step_size(Δφ_L::Float64, data::SolverData, options::Options)
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
    min_step_size = max(min_step_size, eps(Float64))
    return min_step_size
end

function expected_decrease_cost(policy::PolicyData, problem::ProblemData, step_size::Float64)
    Δφ_L = 0.0
    Δφ_Q = 0.0
    N = problem.horizon
    Qu = policy.action_value.gradient_action
    Quu = policy.action_value.hessian_action_action
    
    for k = N-1:-1:1
        Δφ_L += dot(Qu[k], policy.ku[k])
        Δφ_Q += 0.5 * dot(policy.ku[k], Quu[k], policy.ku[k])
    end
    return Δφ_L * step_size, Δφ_Q * step_size^2
end

