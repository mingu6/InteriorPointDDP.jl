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
    # J_prev = data.objective[1]

    # TODO: ARMIJO LINE SEARCH
    # # gradient of Lagrangian
    # lagrangian_gradient!(data, policy, problem)

    # if line_search == :armijo
    #     trajectory_sensitivities(problem, policy, data) ## used to calculate problem.trajectory
    #     delta_grad_product = data.gradient' * problem.trajectory
    # else
    #     delta_grad_product = 0.0
    # end

    # line search with rollout
    data.step_size[1] = 1.0
    iteration = 1

    feasible = options.feasible
    μ_j = data.μ_j
    constr_data = problem.objective.costs.constraint_data
    constraints = constr_data.constraints
    dynamics = problem.model.dynamics
    violations = constr_data.violations
    slacks = constr_data.slacks

    while data.step_size[1] >= min_step_size # check whether we still want it to be this
        iteration > max_iterations && (verbose && (@warn "forward pass failure"), break)

        J = Inf
        
        data.status[1] = rollout!(policy, problem, feasible, μ_j, step_size=data.step_size[1])

        if data.status[1] # not failed
            cost!(data, problem, mode=:current)[1] # calls cost in methods.jl, which calls cost in interior_point.jl, saves result in data.objective[1]
            J = data.objective[1]

            if options.feasible
                # update constraints with computed values in the cache (from rollout_feasible!)
                constraint!(constr_data.violations, constr_data.inequalities, constraints, problem.states, problem.actions, problem.parameters)
                if any(vcat(constr_data.inequalities...) .> 0.)
                    for (i, el) = enumerate(constr_data.inequalities)
                        if any(el .> 0.)
                            display(i)
                            display(el)
                        end
                    end
                end
                logcost = J - μ_j * sum(log.(vcat((-1 .* constr_data.inequalities)...)))
                err = data.optimality_error
            else
                # infeasible
                logcost = J - μ_j * sum(log.(vcat(slacks...)))
                # update constraint values with new states and actions
                constraint!(constr_data.violations, constr_data.inequalities, constraints, problem.states, problem.actions, problem.parameters)
                # err = max(options.optimality_tolerance, norm(vcat(constr_data.violations...) + vcat(slacks...), 1))
                err = data.optimality_error
            end

            candidate = [logcost, err]
            if all(candidate .>= data.filter)
                data.status[1] = false
                data.step_size[1] *= 0.5
                iteration += 1
                continue
            else
                # logcost and err are both lower than filter
                # update variables
                update_nominal_trajectory!(problem, options.feasible) # updates states, actions, duals, and slack vars
                data.objective[1] = J # update cost
                data.status[1] = true # update status
                data.logcost = logcost
                data.optimality_error = err
                # constraints are updated above
                data.filter = [data.logcost, data.optimality_error]  # update filter
                break
            end
        else
            data.step_size[1] *= 0.5
            iteration += 1
            continue
        end
    end

    if data.step_size[1] < min_step_size
        data.status[1] = false
        data.step_size[1] = 0
        verbose && (@warn "line search failure")
    end
end