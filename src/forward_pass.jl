function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options;
    line_search=:armijo,
    min_step_size=1.0e-5,
    c1=1.0e-4,
    c2=0.9,
    max_iterations=25,
    verbose=false)

    # reset solver status
    data.status[1] = false # this is the same as failed in MATLAB

    # previous cost
    J_prev = data.objective[1]

    # gradient of Lagrangian
    lagrangian_gradient!(data, policy, problem) ## TODO: explore what this changes

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

    while data.step_size[1] >= min_step_size
        iteration > max_iterations && (verbose && (@warn "forward pass failure"), break)

        J = Inf
        

        succ = rollout!(policy, problem, feasible, step_size=data.step_size[1])

        data.status[1] = succ

        if succ
            J = cost!(data, probem, mode=:current)[1] # dunno what current does
            if options.feasible
                logcost = J - data.perturbation * sum(log(reshape(- cnew, 1, :))) # do cnew

        else
            continue
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

    end
    data.step_size[1] < min_step_size && (verbose && (@warn "line search failure"))
end

# function strict_constraint_satisfaction(constraints::ConstraintsData)
#     is_satisfied = true
    
#     # set tau
#     τ = max(0.99, 1 - constraints.μ)

#     # Approach 1: checks if any violation is greater than or equal to 0
#     H = len(constraints.constraints)
#     for t = 1:H-1
#         for violation in constraints.violations[t]
#             if violation >= 0
#                 is_satisfied = false
#                     break
#             end
#         end
#     end

#     # TODO: Approach 2: manually check constraints using indices_inequality
#     return is_satisfied
# end


# function inequality_dual_positivity(constraints:: ConstraintsData)
#     is_satisfied = true
#     H = len(constraints.constraints)
#     for t = 1:H-1
#         for dual_var in constraints.ineq[t]
#             # check if s_t > 0
#             if !(dual_var > 0) 
#                 is_satisfied = false
#                     break
#             end
#         end
#     end
#     return is_satisfied
# end