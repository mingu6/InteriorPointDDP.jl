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
            Δφ = expected_decrease_barrier_obj(policy, problem, data.μ, options.feasible)
            min_step_size = estimate_min_step_size(Δφ, data, options)
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
            ((-Δφ) ^ options.s_φ * α > options.δ * θ_prev ^ options.s_θ)
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * α * Δφ
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

function estimate_min_step_size(Δφ::Float64, data::SolverData, options::Options)
    # compute minimum step size based on linear models of step acceptance conditions
    θ_min = data.min_primal_1
    θ = data.primal_1_curr
    γ_θ = options.γ_θ
    γ_α = options.γ_α
    γ_φ = options.γ_φ
    s_θ = options.s_θ
    s_φ = options.s_φ
    δ = options.δ
    if Δφ < 0.0 && θ <= θ_min
        min_step_size = min(γ_θ, -γ_φ * θ / Δφ, δ * θ ^ s_θ / (-Δφ) ^ s_φ)
    elseif Δφ < 0.0 && θ > θ_min
        min_step_size = min(γ_θ, -γ_φ * θ / Δφ)
    else
        min_step_size = γ_θ
    end
    min_step_size *= γ_α
    min_step_size = max(min_step_size, eps(Float64))
    return min_step_size
end

function expected_decrease_barrier_obj(policy::PolicyData, problem::ProblemData, μ::Float64, feasible::Bool)
    Δφ = 0.0  # expected barrier cost decrease
    N = problem.horizon
    
    constr_data = problem.constraints
    g = constr_data.nominal_inequalities
    y = constr_data.nominal_slacks
    
    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Jacobian of inequality constraints 
    gx = constr_data.jacobian_state
    gu = constr_data.jacobian_action
    # Cost gradients
    lx = problem.costs.gradient_state
    lu = problem.costs.gradient_action
    
    policy.x_tmp[N] .= lx[N]
    
    for k = N-1:-1:1
        if feasible
            policy.u_tmp[k] .= lu[k]
            mul!(policy.u_tmp[k], gu[k]', 1.0 ./ g[k], -μ, 1.0)
            mul!(policy.u_tmp[k], fu[k]', policy.x_tmp[k+1], 1.0, 1.0)
            policy.x_tmp[k] .= lx[k]
            mul!(policy.x_tmp[k], gx[k]', 1.0 ./ g[k], -μ, 1.0)
            mul!(policy.x_tmp[k], fx[k]', policy.x_tmp[k+1], 1.0, 1.0)
        else
            policy.u_tmp[k] .= lu[k]
            mul!(policy.u_tmp[k], fu[k]', policy.x_tmp[k+1], 1.0, 1.0)
            policy.x_tmp[k] .= lx[k]
            mul!(policy.x_tmp[k], fx[k]', policy.x_tmp[k+1], 1.0, 1.0)
            # TODO: infeasible needs to account for slack variables
            policy.s_tmp[k] .= -1.0 ./ y[k]
        end
        Δφ += dot(policy.u_tmp[k], policy.ku[k])
        # Δφ += dot(policy.s_tmp[k], policy.ky[k])
    end
    return Δφ
end

