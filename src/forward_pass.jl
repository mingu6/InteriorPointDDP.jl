function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; verbose=false)
    data.status[1] = true
    data.step_size[1] = 1.0
    l = 1  # line search iteration
    min_step_size = -Inf

    μ_j = data.μ_j
    τ = max(options.τ_min, 1 - μ_j)  # fraction-to-boundary parameter
    constr_data = problem.constraints
    
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks
    c = constr_data.inequalities
    c̄ = constr_data.nominal_inequalities
    s̄ = constr_data.nominal_ineq_duals
    ȳ = constr_data.nominal_slacks
    
    constr_violation = 0.0
    barrier_obj = 0.0
    switching = false
    armijo = false
    Δφ = 0.0

    while data.step_size[1] >= min_step_size # check whether we still want it to be this
        # generate proposed increment. reduces step size if NaN or Inf encountered
        try
            rollout!(policy, problem, options.feasible, step_size=data.step_size[1])
        catch
            data.step_size[1] *= 0.5
            l += 1
            continue
        end
        # calibrate minimum step size based on linear approximation of optimality conditions
        if l == 1
            Δφ = expected_decrease_barrier_obj(policy, problem, μ_j, options.feasible)
            if Δφ < 0.0 && data.constr_viol_norm <= data.θ_min
                min_step_size = min(options.γ_θ, -options.γ_φ * data.constr_viol_norm / Δφ,
                    options.δ * data.constr_viol_norm ^ options.s_θ / (-Δφ) ^ options.s_φ)
            elseif Δφ < 0.0 && data.constr_viol_norm > data.θ_min
                min_step_size = min(options.γ_θ, -options.γ_φ * data.constr_viol_norm / Δφ)
            else
                min_step_size = options.γ_θ
            end
            min_step_size *= options.γ_α
            min_step_size = max(min_step_size, eps(Float64))  # machine eps lower bound
        end
        
        # check positivity using fraction-to-boundary condition on dual/slack variables (or constraints for feasible IPDDP)
        constraint!(constr_data, problem.states, problem.actions, problem.parameters)
        data.status[1] = check_fraction_boundary(s, s̄, problem, τ, false)
        data.status[1] = data.status[1] && (options.feasible ? check_fraction_boundary(c, c̄, problem, τ, true) 
            : check_fraction_boundary(y, ȳ, problem, τ, false))
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # 1-norm of constraint violation of proposed step for filter (infeasible IPDDP only)
        constr_violation = options.feasible ? 0. : constraint_violation_1norm(constr_data)  
        
        # evaluate objective function of barrier problem to assess quality of iterate
        barrier_obj = barrier_objective!(problem, data, options.feasible, mode=:current)
        
        # check acceptability to filter A-5.4 IPOPT
        for pt in data.filter
            if constr_violation >= pt[1] && barrier_obj >= pt[2]  # violation should stay 0. for fesible IPDDP
                data.status[1] = false
                break
            end
        end
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # additional checks for validity, e.g., switching condition + armijo or sufficient improvement w.r.t. filter
        # NOTE: if feasible, constraint violation not considered. just check armijo condition to accept trial point
        switching = ((Δφ < 0.0) && 
            ((-Δφ) ^ options.s_φ * data.step_size[1] > options.δ * data.constr_viol_norm ^ options.s_θ))
        # for armijo condition, add adjustment to account for round-off error
        armijo = barrier_obj - data.barrier_obj - 10. * eps(Float64) * abs(data.barrier_obj) <= options.η_φ * data.step_size[1] * Δφ
        if (constr_violation <= data.θ_min) && switching
            data.status[1] = armijo
        else
            # sufficient progress conditions
            suff = !options.feasible ? (constr_violation <= (1. - options.γ_θ) * data.constr_viol_norm) : false
            suff = suff || (barrier_obj <= data.barrier_obj - options.γ_φ * data.constr_viol_norm)
            !suff && (data.status[1] = false)
        end
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        break
    end
    !data.status[1] && (verbose && (@warn "Line search failed to find a suitable iterate"))
    return constr_violation, barrier_obj, switching, armijo
end

function check_fraction_boundary(s, s̄, problem::ProblemData, τ::Float64, flip::Bool)
    H = problem.horizon
    constr_data = problem.constraints
    if !flip
        for t = 1:H
            num_constraint = constr_data.constraints[t].num_inequality
            for i = 1:num_constraint
                s[t][i] < (1. - τ) *  s̄[t][i] && return false
            end
        end
    else
        for t = 1:H
            num_constraint = constr_data.constraints[t].num_inequality
            for i = 1:num_constraint
                s[t][i] > (1. - τ) *  s̄[t][i] && return false
            end
        end
    end
    return true
end

function expected_decrease_barrier_obj(policy::PolicyData, problem::ProblemData, μ_j::Float64, feasible::Bool)
    Δφ = 0.0  # expected barrier cost decrease
    Qu = policy.action_value.gradient_action
    N = length(Qu) + 1
    
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
            mul!(policy.u_tmp[k], gu[k]', 1.0 ./ g[k], -μ_j, 1.0)
            mul!(policy.u_tmp[k], fu[k]', policy.x_tmp[k+1], 1.0, 1.0)
            policy.x_tmp[k] .= lx[k]
            mul!(policy.x_tmp[k], gx[k]', 1.0 ./ g[k], -μ_j, 1.0)
            mul!(policy.x_tmp[k], fx[k]', policy.x_tmp[k+1], 1.0, 1.0)
        else
            policy.u_tmp[k] .= lu[k]
            mul!(policy.u_tmp[k], fu[k]', policy.x_tmp[k+1], 1.0, 1.0)
            policy.x_tmp[k] .= lx[k]
            mul!(policy.x_tmp[k], fx[k]', policy.x_tmp[k+1], 1.0, 1.0)
        end
        Δφ += dot(policy.u_tmp[k], policy.ku[k])
    end
    return Δφ
end

function rescale_duals!(s, cy, problem::ProblemData, μ_j::Float64, options::Options)
    H = problem.horizon
    constr_data = problem.constraints
    κ_Σ = options.κ_Σ 
    if options.feasible
        for t = 1:H
            num_constraint = constr_data.constraints[t].num_inequality
            for i = 1:num_constraint
                s[t][i] = max(min(s[t][i], -κ_Σ * μ_j / cy[t][i]), -μ_j / (κ_Σ *  cy[t][i]))
            end
        end
    else
        for t = 1:H
            num_constraint = constr_data.constraints[t].num_inequality
            for i = 1:num_constraint
                s[t][i] = max(min(s[t][i], κ_Σ * μ_j / cy[t][i]), μ_j / (κ_Σ *  cy[t][i]))
            end
        end
    end
end
