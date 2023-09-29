function forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, constraints::ConstraintsData;
    line_search=:armijo,
    min_step_size=1.0e-5,
    c1=1.0e-4,
    c2=0.9,
    max_iterations=25,
    verbose=false)

    # reset solver status
    data.status[1] = false

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
    while data.step_size[1] >= min_step_size
        iteration > max_iterations && (verbose && (@warn "forward pass failure"), break)

        J = Inf

        # try
        is_satisfied = rollout!(policy, problem, constraints,
            step_size=data.step_size[1])
        J = cost!(data, problem, 
            mode=:current)[1]
        # catch
        #     if verbose
        #         @warn "rollout failure"
        #         @show norm(data.gradient)
        #     end
        # end
        
        # check Armijo and strict constraint satisfaction and that s dual are all positive
        if (J <= J_prev + c1 * data.step_size[1] * delta_grad_product && is_satisfied)
            # update nominal
                update_nominal_trajectory!(problem, constraints)
                data.objective[1] = J
                data.status[1] = true
                break
        else
            data.step_size[1] *= 0.5
            iteration += 1
        end
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