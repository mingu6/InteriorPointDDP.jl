function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; verbose=false)
    data.status = true
    data.step_size = 1.0
    constr_data = problem.constraints
    l = 0  # line search iteration counter
    min_step_size = -Inf

    μ = data.μ
    τ = max(options.τ_min, 1.0 - μ)
    
    Δφ = 0.0

    while data.step_size >= min_step_size # check whether we still want it to be this
        α = data.step_size
        try
            rollout!(policy, problem, options.feasible, step_size=α)
        catch
            # reduces step size if NaN or Inf encountered
            data.step_size *= 0.5
            l += 1
            continue
        end
        constraint!(constr_data, problem.states, problem.actions, problem.parameters)

        Δφ = expected_decrease_barrier_obj(policy, problem, data.μ, options.feasible)
        min_step_size == -Inf && (min_step_size = estimate_min_step_size(Δφ, data, options))
        
        data.status = check_fraction_boundary(constr_data, τ, options.feasible)
        !data.status && (data.step_size *= 0.5, l += 1, continue)
        
        # 1-norm of constraint violation of proposed step for filter (infeasible IPDDP only)
        constr_violation = options.feasible ? 0. : constraint_violation_1norm(constr_data)  
        
        # evaluate objective function of barrier problem to assess quality of iterate
        φ = barrier_objective!(problem, data, options.feasible, mode=:current)
        φ_prev = data.barrier_obj_curr
        
        # check acceptability to filter A-5.4 IPOPT
        for pt in data.filter
            if constr_violation >= pt[1] && φ >= pt[2]  # violation should stay 0. for fesible IPDDP
                data.status = false
                break
            end
        end
        # println("filter ", data.status, " ", data.filter)
        !data.status && (data.step_size *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # additional checks for validity, e.g., switching condition + armijo or sufficient improvement w.r.t. filter
        # NOTE: if feasible, constraint violation not considered. just check armijo condition to accept trial point
        data.switching = (Δφ < 0.0) && 
            ((-Δφ) ^ options.s_φ * α > options.δ * data.primal_1_curr ^ options.s_θ)  # TODO: rename primal_1_curr to θ here
        # for armijo condition, add adjustment to account for round-off error
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * α * Δφ
        if (constr_violation <= data.min_primal_1) && data.switching
            data.status = data.armijo_passed
            # println(data.armijo_passed, " ", data.switching, " ", Δφ)
        else
            # sufficient progress conditions
            suff = !options.feasible ? (constr_violation <= (1. - options.γ_θ) * data.primal_1_curr) : false
            suff = suff || (φ <= φ_prev - options.γ_φ * data.primal_1_curr)
            !suff && (data.status = false)
            # println(data.armijo_passed, " ", data.switching, " ", suff, " ", Δφ)
        end
        !data.status && (data.step_size *= 0.5, l += 1, continue)  # failed, reduce step size
        # step size accepted, update performance of new iterate as current
        data.barrier_obj_next = φ
        data.primal_1_next = constr_violation
        break
    end
    !data.status && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function check_fraction_boundary(constr_data::ConstraintsData, τ::Float64, feasible::Bool)
    constraints = constr_data.constraints
    N = length(constraints)
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks
    c̄ = constr_data.nominal_inequalities
    s̄ = constr_data.nominal_ineq_duals
    ȳ = constr_data.nominal_slacks
    for k = 1:N
        for i = constr_data.constraints[k].indices_inequality
            fail = s[k][i] < (1. - τ) *  s̄[k][i]
            fail = fail || (feasible ? c[k][i] > (1. - τ) *  c̄[k][i] : y[k][i] < (1. - τ) *  ȳ[k][i])
            fail && return false
        end
    end
    return true
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
    min_step_size = max(min_step_size, eps(Float64))  # machine eps lower bound
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
        end
        Δφ += dot(policy.u_tmp[k], policy.ku[k])
    end
    return Δφ
end

