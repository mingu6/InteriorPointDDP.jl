function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; verbose=false)
    constr_data = problem.constraints
    
    data.l = 0  # line search iteration counter
    data.status = true
    data.step_size = 1.0
    min_step_size = -Inf
    Δφ = 0.0
    μ = data.μ
    τ = max(options.τ_min, 1.0 - μ)

    while data.step_size >= min_step_size # check whether we still want it to be this
        α = data.step_size
        try
            rollout!(policy, problem, options.feasible, step_size=α)
        catch
            # reduces step size if NaN or Inf encountered
            data.step_size *= 0.5
            data.l += 1
            continue
        end
        constraint!(problem, mode=:current)
        
        data.status = check_fraction_boundary(constr_data, τ, options.feasible)
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)

        if min_step_size == -Inf
            Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, data.step_size)
            min_step_size = estimate_min_step_size(Δφ_L, data, options)
            Δφ = Δφ_L + Δφ_Q
        end
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = options.feasible ? 0. : constraint_violation_1norm(constr_data, mode=:current)
        φ = barrier_objective!(problem, data, options.feasible, mode=:current)
        θ_prev = data.primal_1_curr
        φ_prev = data.barrier_obj_curr
        
        # check acceptability to filter
        data.status = !any(x -> all([θ, φ] .>= x), data.filter)
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        # check for sufficient decrease conditions for the barrier objective/constraint violation
        data.switching = (Δφ < 0.0) && 
            ((-Δφ) ^ options.s_φ * α^(1-options.s_φ)  > options.δ * θ_prev ^ options.s_θ)
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * Δφ
        if (θ <= data.min_primal_1) && data.switching
            data.status = data.armijo_passed  #  sufficient decrease of barrier objective
        else
            suff = !options.feasible ? (θ <= (1. - options.γ_θ) * θ_prev) : false
            suff = suff || (φ <= φ_prev - options.γ_φ * θ_prev)
            !suff && (data.status = false)
        end
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        data.barrier_obj_next = φ
        data.primal_1_next = θ
        break
    end
    !data.status && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function check_fraction_boundary(constr_data::ConstraintsData, τ::Float64, feasible::Bool)
    constraints = constr_data.constraints
    N = length(constraints)
    c̄, s̄, ȳ = dual_trajectories(constr_data, mode=:nominal)
    c, s, y = dual_trajectories(constr_data, mode=:current)
    check_feasible = k -> any(s[k] .< (1. - τ) .*  s̄[k]) || any(c[k] .> (1. - τ) .*  c̄[k])
    check_infeasible = k -> any(s[k] .< (1. - τ) .*  s̄[k]) || any(y[k] .< (1. - τ) .*  ȳ[k])
    check_fn = feasible ? check_feasible : check_infeasible
    return !any(check_fn, 1:N)
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

