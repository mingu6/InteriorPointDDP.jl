function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options;
    line_search=:armijo,
    min_step_size=1.0e-5,
    c1=1.0e-4,
    c2=0.9,
    max_iterations=25,
    verbose=false)

    # reset solver status
    data.status[1] = true # this is the same as (not) failed in MATLAB

    # # previous cost
    # J_prev = data.objective[1]

    # gradient of Lagrangian
    lagrangian_gradient!(data, policy, problem)

    if line_search == :armijo
        trajectory_sensitivities(problem, policy, data) ## used to calculate problem.trajectory
        delta_grad_product = data.gradient' * problem.trajectory
    else
        delta_grad_product = 0.0
    end

    # line search with rollout
    data.step_size[1] = 1.0
    iteration = 1

    feasible = options.feasible
    perturbation = data.pertubation
    constr_data = problem.objective.costs.constraint_data
    dynamics = problem.model.dynamics
    violations = constr_data.violations

    while data.step_size[1] >= min_step_size # check whether we still want it to be this
        iteration > max_iterations && (verbose && (@warn "forward pass failure"), break)

        J = Inf
        
        data.status[1] = rollout!(policy, problem, feasible, perturbation, step_size=data.step_size[1])

        if data.status[1] # not failed
            cost!(data, probem, mode=:current)[1] # calls cost in methods.jl, which calls cost in interior_point.jl, saves result in data.objective[1]
            J = data.objective[1]

            if options.feasible
                # update constraints with computed values in the cache (from rollout_feasible!)
                for t = 1:length(dynamics)
                    @views violations[t] .= constr_data[t].evaluate_cache
                    # take inequalities and package them together
                    @views constr_data.inequalities[t] .= constr_data[t].evaluate_cache[constr_data[t].indices_inequality] # cool indexing trick
                end
                logcost = J - perturbation * sum(log(-1.0 .* reshape(violations[t], 1, :))) 
                err = 0
            else
                # infeasible
                logcost = J - perturbation * sum(log(reshape(constr_data.slacks, 1, :)))
                # update constraint values with new states and actions
                constraint!(constr_data.violations, constr_data.inequalities, constr_data.constraints, problem.states, problem.actions, problem.parameters)
                err = max(options.objective_tolerance, norm(reshape(constr_data.violations .+ constr_data.slacks ,1,:), 1))
            end

            candidate = [logcost, err]
            if any(all(candidate .>= data.filter, dims=1))
                data.status[1] = false
                data.step_size[1] *= 0.5
                iteration += 1
                continue
            else
                # logcost and err are both lower than filter
                # update variables
                update_nominal_trajectory!(problem) # updates states, actions, duals, and slack vars
                data.objective[1] = J # update cost
                data.status[1] = true # update status
                data.logcost = logcost
                data.err = err
                # constraints are updated above
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



        # if succ # if rollout is successful
        #     J = cost!(data, problem, 
        #     mode=:current)[1]
        
        #     # check Armijo and strict constraint satisfaction and that s dual are all positive
        #     if (J <= J_prev + c1 * data.step_size[1] * delta_grad_product)
        #     # update nominal
        #         update_nominal_trajectory!(problem)
        #         data.objective[1] = J
        #         data.status[1] = true
        #         break
        #     else
        #         data.step_size[1] *= 0.5
        #         iteration += 1
        #     end
        # else
        #     data.step_size[1] *= 0.5
        #     iteration += 1
        #     continue
        #     continue
        # end

        # if data.status[1] == false
        #     continue
        # else
        #     for t = 1:options.horizon