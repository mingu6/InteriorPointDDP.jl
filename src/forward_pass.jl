function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; verbose=false)
    data.l = 0  # line search iteration counter
    data.status = 0
    data.step_size = 1.0
    Δφ = 0.0
    μ = data.μ
    τ = max(options.τ_min, 1.0 - μ)

    θ_prev = data.primal_1_curr
    φ_prev = data.barrier_obj_curr
    θ = θ_prev

    Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, data.step_size)
    Δφ = Δφ_L + Δφ_Q
    min_step_size = estimate_min_step_size(Δφ_L, data, options)

    while data.step_size >= min_step_size
        # # filter reset heuristic, if filter is blocking large steps for several iterations, reset
        # if data.t == 6 && data.status == 6 && data.max_primal_1 > θ / 10.0
        #     data.max_primal_1 *= 0.1
        #     reset_filter!(data)
        #     data.t = 0
        # end

        if data.l == 1 && data.status != 0 && θ >= θ_prev
            soc_status = second_order_correction!(policy, problem, data, options, τ)
            !soc_status && (data.status = 6)
            data.status == 0 && break

            data.step_size *= 0.5
            data.l += 1
            continue
        end

        α = data.step_size
        try
            rollout!(policy, problem, τ, step_size=α; mode=:main)
        catch
            # reduces step size if NaN or Inf encountered
            data.step_size *= 0.5
            continue
        end
        constraint!(problem, data.μ; mode=:current)
        
        data.status = check_fraction_boundary(problem, τ)
        # println("boundary failed")
        data.status != 0 && (data.step_size *= 0.5, continue)

        Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, α; mode=:main)
        Δφ = Δφ_L + Δφ_Q
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = constraint_violation_1norm(problem, mode=:current)
        φ = barrier_objective!(problem, data, mode=:current)
        
        # check acceptability to filter
        data.status = !any(x -> all([θ, φ] .>= x), data.filter) ? 0 : 3
        # println("filter ", data.k, " ", data.status, " ", α, " ", θ, " ", φ, " ", μ)
        # println(data.filter)
        data.status != 0 && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        # check for sufficient decrease conditions for the barrier objective/constraint violation
        data.switching = (Δφ < 0.0) && 
            ((-Δφ) ^ options.s_φ * α^(1-options.s_φ)  > options.δ * θ_prev ^ options.s_θ)
        # println("switch ", data.switching, " ", θ_prev, " ", -Δφ, " ", min_step_size)
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * Δφ
        # println("armijo ", data.armijo_passed, " ", θ <= data.min_primal_1, " ", φ - φ_prev, " ", Δφ, " ", data.step_size)
        if (θ <= data.min_primal_1) && data.switching
            data.status = data.armijo_passed ? 0 : 4  #  sufficient decrease of barrier objective
        else
            suff = (θ <= (1. - options.γ_θ) * θ_prev) || (φ <= φ_prev - options.γ_φ * θ_prev)
            # println("suff ", θ, " ", (1. - options.γ_θ) * θ_prev, " ", φ, " ", φ_prev - options.γ_φ * θ_prev)
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

function check_fraction_boundary(problem::ProblemData, τ::Float64)
    N = problem.horizon
    x, u, _, il, iu = primal_trajectories(problem, mode=:current)
    _, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, _, il̄, iū = primal_trajectories(problem, mode=:nominal)
    _, vl̄, vū = dual_trajectories(problem, mode=:nominal)
    constraints = problem.constr_data.constraints

    status = 0
    for k = 1:N-1
        if any((il[k] .< (1. - τ) .*  il̄[k]) .* .!isinf.(il̄[k]))
            # println(k, " il ", il[k], " ", (1. - τ + τ) .*  il̄[k])
            status = 2
            break
        elseif any((iu[k] .< (1. - τ) .*  iū[k]) .* .!isinf.(iū[k]))
            # println(k, " iu ", iu[k], " ", (1. - τ + τ) .*  iū[k])
            status = 2
            break
        elseif any((vl[k] .< (1. - τ) .*  vl̄[k]) .* .!isinf.(vl̄[k]))
            # println(k, " vl ", vl[k], " ", (1. - τ + τ) .*  vl̄[k])
            status = 2
            break
        elseif any((vu[k] .< (1. - τ) .*  vū[k]) .* .!isinf.(vū[k]))
            status = 2
            # println(k, " vu ", vu[k], " ", (1. - τ + τ) .*  vū[k])
            break
        end
    end
    return status
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

function expected_decrease_cost(policy::PolicyData, problem::ProblemData, step_size::Float64; mode=:main)
    Δφ_L = 0.0
    Δφ_Q = 0.0
    N = problem.horizon
    Qu = policy.action_value.gradient_action
    Quu = policy.action_value.hessian_action_action
    gains = mode == :main ? policy.gains_main : policy.gains_soc
    
    for k = N-1:-1:1
        Δφ_L += dot(Qu[k], gains.ku[k])
        Δφ_Q += 0.5 * dot(gains.ku[k], Quu[k], gains.ku[k])
    end
    return Δφ_L * step_size, Δφ_Q * step_size^2
end

