using LinearAlgebra

function backward_pass!(policy::PolicyData, problem::ProblemData, solver_data::SolverData,
    options::Options, mode=:nominal)

    # Horizon 
    N = length(problem.states)
    
    # Constraint Data 
    constraints = problem.objective.costs.constraint_data

    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    # fxx = problem.model.hessian_state_state
    # fxu = problem.model.hessian_state_action
    # fuu = problem.model.hessian_action_action
    # Jacobian constraints 
    Qsx = constraints.jacobian_state
    Qsu = constraints.jacobian_action
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
    s = constraints.ineq_duals

    # terminal value function
    # TODO: Extension: Implement terminal constraints 
    Vxx[N] .= qxx[N]
    Vx[N] .= qx[N]

    for t = N-1:-1:1
        # Qx[t] .= qx[t] + (Qsx[t]' * s[t]) + (fx[t]' * Vx[t+1])
        mul!(policy.x_tmp[t], transpose(Qsx[t]), s[t])
        mul!(Qx[t], transpose(fx[t]), Vx[t+1])
        Qx[t] .+= qx[t] + policy.x_tmp[t]
        # Qx[t] .= qx[t] + (Qsx[t]' * s[t]) + (fx[t]' * Vx[t+1])

        Qu[t] .= qu[t] + (Qsu[t]' * s[t]) + (fu[t]' * Vx[t+1])
        mul!(policy.u_tmp[t], transpose(Qsu[t]), s[t])
        mul!(Qu[t], transpose(fu[t]), Vx[t+1])
        Qu[t] .+= qu[t] + policy.u_tmp[t]
        # Qu[t] .= qu[t] + (Qsu[t]' * s[t]) + (fu[t]' * Vx[t+1])

        # Qxx[t] .= qxx[t] + ((fx[t]' * Vxx[t+1]) * fx[t])
        mul!(policy.xx̂_tmp[t], transpose(fx[t]), Vxx[t+1])
        mul!(Qxx[t], policy.xx̂_tmp[t], fx[t])
        Qxx[t] .+= qxx[t]
        # Qxx[t] .= qxx[t] + ((fx[t]' * Vxx[t+1]) * fx[t])

        # Quu[t] .= quu[t] + ((fu[t]' * Vxx[t+1]) * fu[t])
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Quu[t], policy.ux̂_tmp[t], fu[t])
        Quu[t] .+= quu[t]
        # Quu[t] .= quu[t] + ((fu[t]' * Vxx[t+1]) * fu[t])

        # Qux[t] .= qux[t] + ((fu[t]' * Vxx[t+1]) * fx[t])
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Qux[t], policy.ux̂_tmp[t], fx[t])
        Qux[t] .+= qux[t]
        # Qux[t] .= qux[t] + ((fu[t]' * Vxx[t+1]) * fx[t])

        # Future TODO 
        # 1) Call direct factorisation instead of cholesky
        if options.feasible
            # Compute gains
            feedback_policy_feasible!(policy, problem, options, solver_data.μ_j, t)
            value_funcs_feasible!(policy, constraints, solver_data.μ_j, t)
        else
            # Compute gains 
            r = compute_gains_and_update_infeasible!(policy, problem, solver_data, options, t)
        end

        update_value_function!(policy, t)
    end
end

function feedback_policy_feasible!(policy::PolicyData, problem::ProblemData, options::Options, μ_j::Float64, t::Int)
    # Constraint Data 
    constraints = problem.objective.costs.constraint_data
    # Feedback gains (linear feedback policy)
    Ku = policy.Ku
    ku = policy.ku
    Ks = policy.Ks
    ks = policy.ks
    kkt_soln = policy.kkt_soln_tmp
    # Action-Value function approximation
    Qu = policy.action_value.gradient_action
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Constraint jacobians
    Qsx = constraints.jacobian_state
    Qsu = constraints.jacobian_action
    # Cost hessian 
    quu = problem.objective.hessian_action_action
    
    c = constraints.inequalities
    s = constraints.ineq_duals
    r = s[t] .* c[t] .+ μ_j
    cinv = 1.0 ./ c[t]
    s_cinv = s[t] .* cinv

    # Iteratively bump regularisation
    code = 0
    reg = options.start_reg
    
    # for reg=options.start_reg:options.reg_step:options.end_reg
    while reg < options.end_reg
        policy.uu_tmp[t] .= Quu[t] + quu[t] * (1.6^reg - 1) .- Qsu[t]' * (s_cinv .* Qsu[t])
        (_, code) = LAPACK.potrf!('U', policy.uu_tmp[t])
        if code > 0
            reg += options.reg_step
            continue
        end
        kkt_soln[t] .= -hcat(Qu[t] - Qsu[t]' * (cinv .* r), Qux[t] - Qsu[t]' * (s_cinv .* Qsx[t]))
        LAPACK.potrs!('U', policy.uu_tmp[t], kkt_soln[t])
        break
    end
    if code > 0  # cycled through all regularization values and still not PD
        error("Regularisation failed for all values")
    end
    
    options.reg = reg

    # Update gains
    ku[t] = kkt_soln[t][:, 1]
    Ku[t] = kkt_soln[t][:, 2:end]
    ks[t] = -cinv .* (r .+ s[t] .* Qsu[t] * ku[t])
    Ks[t] = -s_cinv .* (Qsx[t] .+ Qsu[t] * Ku[t])
end


function value_funcs_feasible!(policy::PolicyData, constraints::ConstraintsData, μ_j::Float64, t::Int)
    # Gains (linear feedback policy)
    Ku = policy.Ku
    ku = policy.ku
    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Constraint jacobians
    Qsx = constraints.jacobian_state
    Qsu = constraints.jacobian_action
    
    # Value function approximations
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Gains values
    Ku = policy.Ku
    ku = policy.ku
    
    c = constraints.inequalities
    s = constraints.ineq_duals
    r = s[t] .* c[t] .+ μ_j
    cinv = 1.0 ./ c[t]
    s_cinv = s[t] .* cinv
    
    # Update quadratic Q-function (state-action) approximation
    Quu[t] .-= Qsu[t]' * (s_cinv .* Qsu[t])
    Qux[t] .-= transpose(Qsx[t]' * (s_cinv .* Qsu[t]))
    Qxx[t] .-= Qsx[t]' * (s_cinv .* Qsx[t])
    Qu[t] .-= Qsu[t]' * (cinv .* r)
    Qx[t] .-= Qsx[t]' * (cinv .* r)
    
    # Update quadratic value function approximation
    # Vxx[t] .=  Qxx[t] + (K[t]' * (Quu[t] * K[t])) + (K[t]' * Qux[t]) + (Qux[t]' * K[t])
    mul!(policy.ux_tmp[t], Quu[t], Ku[t])
    mul!(Vxx[t], transpose(Ku[t]), policy.ux_tmp[t])
    mul!(Vxx[t], transpose(Ku[t]), Qux[t], 1.0, 1.0) # apply appropriate scaling 
    mul!(Vxx[t], transpose(Qux[t]), Ku[t], 1.0, 1.0) # apply appropriate scaling
    Vxx[t] .+= Qxx[t]
    # Vxx[t] .=  Qxx[t] + (Ku[t]' * (Quu[t] * Ku[t])) + (Ku[t]' * Qux[t]) + (Qux[t]' * Ku[t])

    # Vx[t] .=  Qx[t] + (K[t]' * Quu[t] * k[t]) + (K[t]' * Qu[t]) + (Qux[t]' * k[t])
    mul!(policy.u_tmp[t], Quu[t], ku[t])
    mul!(Vx[t], transpose(Ku[t]), policy.u_tmp[t])
    mul!(Vx[t], transpose(Ku[t]), Qu[t], 1.0, 1.0) # apply appropriate scaling 
    mul!(Vx[t], transpose(Qux[t]), ku[t], 1.0, 1.0) # apply appropriate scaling 
    Vx[t] .+= Qx[t]
    # Vx[t] .=  Qx[t] + (Ku[t]' * Quu[t] * ku[t]) + (Ku[t]' * Qu[t]) + (Qux[t]' * ku[t])
end
