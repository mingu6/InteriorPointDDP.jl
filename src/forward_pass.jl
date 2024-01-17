function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options;
    line_search=:armijo,
    min_step_size=1.0e-5,
    c1=1.0e-4,
    c2=0.9,
    max_iterations=100,
    verbose=false)

    # reset solver status
    data.status[1] = true # this is the same as (not) failed in MATLAB

    # # previous cost
    # J_prev = data.costs[1]

    # TODO: ARMIJO LINE SEARCH
    # # gradient of Lagrangian
    # lagrangian_gradient!(data, policy, problem)

    # if line_search == :armijo
    #     trajectory_sensitivities(problem, policy, data) ## used to calculate problem.trajectory
    #     delta_grad_product = data.gradient' * problem.trajectory
    # else
    #     delta_grad_product = 0.0
    # end

    data.step_size[1] = 1.0
    l = 1  # line search iteration

    μ_j = data.μ_j
    τ = max(options.τ_min, 1 - μ_j)  # fraction-to-boundary parameter set 0.99 as parameter in options
    constr_data = problem.constraints
    
    H = length(problem.states)
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks
    c = constr_data.inequalities
    c̄ = constr_data.nominal_inequalities
    s̄ = constr_data.nominal_ineq_duals
    ȳ = constr_data.nominal_slacks

    while data.step_size[1] >= min_step_size # check whether we still want it to be this
        # generate proposed increment
        rollout!(policy, problem, options.feasible, μ_j, step_size=data.step_size[1])
        
        # check positivity using fraction-to-boundary condition on dual/slack variables (or constraints for feasible IPDDP)
        constraint!(constr_data, problem.states, problem.actions, problem.parameters)
        data.status[1] = check_positivity(s, s̄, problem, τ, false)
        data.status[1] = data.status[1] && (options.feasible ? check_positivity(c, c̄, problem, τ, true) : check_positivity(y, ȳ, problem, τ, false))
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # 1-norm of constraint violation of proposed step for filter (infeasible IPDDP only)
        constr_violation = options.feasible ? 0. : constraint_violation_1norm(constr_data)  
        
        # evaluate objective function of barrier problem to assess quality of iterate
        barrier_obj = options.feasible ? barrier_obj_feasible!(problem, data, μ_j) : barrier_obj_infeasible!(problem, data, μ_j)
        
        # check acceptability to filter A-5.4 IPOPT
        ind_replace_filter = 0
        for (i, pt) in enumerate(data.filter)
            if constr_violation >= pt[1] && barrier_obj >= pt[2]  # violation should stay 0. for fesible IPDDP
                data.status[1] = false
                break
            end
            if constr_violation < pt[1] && barrier_obj < pt[2]  # if we update the filter, replace an existing point if possible
                ind_replace_filter = i
            end
        end
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # additional checks for validity, e.g., armijo
        # if failed, then update step size
        
        # accept step!!! update nominal trajectory w/rollout
        update_nominal_trajectory!(problem, options.feasible)
        data.barrier_obj = barrier_obj
        data.constr_viol_norm = constr_violation
        
        # check if filter should be augmented using accepted point
        if true
            new_filter_pt = [constr_violation, barrier_obj]
            ind_replace_filter == 0 ? push!(data.filter, new_filter_pt) : data.filter[ind_replace_filter] = new_filter_pt
        end
        l += 1
        break
    end
    !data.status[1] && (verbose && (@warn "line search failure"))  # to do, just exit!!!
end

function barrier_obj_feasible!(problem::ProblemData, data::SolverData, μ::Float64)
    constr_data = problem.constraints
    barrier_obj = 0.
    for (t, c_t) in enumerate(constr_data.inequalities)
        n_e = constr_data.constraints[t].num_inequality
        for i = 1:n_e
            barrier_obj -= log(-c_t[i])
        end
    end
    barrier_obj *= μ
    cost!(data, problem, mode=:current)[1]
    barrier_obj += data.costs[1]
    return barrier_obj
end

function barrier_obj_infeasible!(problem::ProblemData, data::SolverData, μ::Float64)
    constr_data = problem.constraints
    barrier_obj = 0.
    for (t, y_t) in enumerate(constr_data.slacks)
        n_e = constr_data.constraints[t].num_inequality
        for i = 1:n_e
            barrier_obj -= log(y_t[i])
        end
    end
    barrier_obj *= μ
    cost!(data, problem, mode=:current)[1]
    barrier_obj += data.costs[1]
    return barrier_obj
end

function check_positivity(s, s̄, problem::ProblemData, τ::Float64, flip::Bool)
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
