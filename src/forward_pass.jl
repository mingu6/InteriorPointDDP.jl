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
    constr_data = problem.constraints
    
    H = length(problem.states)
    c = constr_data.inequalities
    y = constr_data.slacks

    while data.step_size[1] >= min_step_size # check whether we still want it to be this
        # generate proposed increment
        data.status[1] = rollout!(policy, problem, options.feasible, μ_j, step_size=data.step_size[1])
        !data.status[1] && (data.step_size[1] *= 0.5, l += 1, continue)  # failed, reduce step size
        
        # compute 1-norm of constraint violation to assess quality of iterate
        constr_violation = 0.  # 1-norm of constraint violation of proposed step (infeasible IPDDP only)
        if !options.feasible
            for t = 1:H
                n_e = constr_data[t].constraints.num_inequality
                for i = 1:n_e
                    constr_violation += abs(c[t][i] + y[t][i])
                end
            end
        end
        
        # evaluate objective function of barrier problem to assess quality of iterate
        # TODO: evaluate constraints!!! maybe do separately outside of rollout. eval constriant after rollout. check non-negativity here
        constraint!(constr_data, problem.states, problem.actions, problem.parameters)
        barrier_obj = options.feasible ? barrier_obj_feasible!(problem, data, μ_j) : barrier_obj_infeasible!(problem, data, μ_j)
        
        # check acceptability to filter A-5.4 IPOPT
        for pt in data.filter
            if constr_violation >= pt[1] && barrier_obj >= pt[2]  # violation should stay 0. for fesible IPDDP
                data.status[1] = false
                break
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
            push!(data.filter, [constr_violation, barrier_obj])
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

