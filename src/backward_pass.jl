using LinearAlgebra

function backward_pass!(policy::PolicyData, 
    problem::ProblemData,
    solver_data::SolverData,
    options::Options,
    mode=:nominal
    )

    # Horizon 
    H = length(problem.states)
    # Constraint Data 
    con_data = problem.objective.costs.constraint_data
    # Errors
    c_err::Float64 = 0
    mu_err::Float64 = 0
    Qu_err::Float64 = 0

    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    # fxx = problem.model.hessian_state_state
    # fxu = problem.model.hessian_state_action
    # fuu = problem.model.hessian_action_action
    # Jacobian constraints 
    Qsx = con_data.jacobian_state
    Qsu = con_data.jacobian_action
    # Cost gradients
    qx = problem.objective.gradient_state
    qu = problem.objective.gradient_action
    # Cost hessians
    qxx = problem.objective.hessian_state_state
    quu = problem.objective.hessian_action_action
    qux = problem.objective.hessian_action_state
    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    # Inequality dual variable 
    s = con_data.ineq_duals

    # terminal value function
    # TODO: Extension: Implement terminal constraints 
    Vxx[H] .= qxx[H]
    Vx[H] .= qx[H]

    for t = H-1:-1:1
        # Qx[t] .= qx[t] + (Qsx[t]' * s[t]) + (fx[t]' * Vx[t+1])
        mul!(policy.x_tmp[t], transpose(Qsx[t]), s[t])
        mul!(Qx[t], transpose(fx[t]), Vx[t+1])
        Qx[t] .+= qx[t] + policy.x_tmp[t]

        # Qu[t] .= qu[t] + (Qsu[t]' * s[t]) + (fu[t]' * Vx[t+1])
        mul!(policy.u_tmp[t], transpose(Qsu[t]), s[t])
        mul!(Qu[t], transpose(fu[t]), Vx[t+1])
        Qu[t] .+= qu[t] + policy.u_tmp[t]

        # Qxx[t] .= qxx[t] + ((fx[t]' * Vxx[t+1]) * fx[t])
        mul!(policy.xx̂_tmp[t], transpose(fx[t]), Vxx[t+1])
        mul!(Qxx[t], policy.xx̂_tmp[t], fx[t])
        Qxx[t] .+= qxx[t]

        # Quu[t] .= quu[t] + ((fu[t]' * Vxx[t+1]) * fu[t])
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Quu[t], policy.ux̂_tmp[t], fu[t])
        Quu[t] .+= quu[t]

        # Qux[t] .= qux[t] + ((fu[t]' * Vxx[t+1]) * fx[t])
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Qux[t], policy.ux̂_tmp[t], fx[t])
        Qux[t] .+= qux[t]

        # Future TODO 
        # 1) Call direct factorisation instead of cholesky
        if options.feasible
            # Compute gains
            r = compute_gains_and_update_feasible!(policy, 
            problem, solver_data, options, t)
        else
            # Compute gains 
            r = compute_gains_and_update_infeasible!(policy,
            problem, solver_data, options, t)
        end

        update_value_function!(policy, t)

        # Optimality Error 
        Qu_err = max(Qu_err, norm(Qu[t], Inf))
        mu_err = max(mu_err, norm(r, Inf))
        if !options.feasible
            constraint_evaluation = con_data.inequalities
            slacks = con_data.slacks
            c_err = max(c_err, norm(constraint_evaluation[t] + slacks[t], Inf))
        end
    end

    options.opterr=max(Qu_err, c_err, mu_err);
end