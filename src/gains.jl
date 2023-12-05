function compute_gains_and_update_feasible!(policy, problem, solver_data, options, t)

    # Constraint Data 
    con_data = problem.objective.costs.constraint_data
    # Gains 
    Ku = policy.Ku
    ku = policy.ku
    Ks = policy.Ks
    ks = policy.ks
    kK = policy.kK_tmp
    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Constraint jacobians
    Qsx = con_data.jacobian_state
    Qsu = con_data.jacobian_action
    # Cost hessian 
    quu = problem.objective.hessian_action_action
    # Constraint evaluation
    constraint_evaluations = con_data.inequalities
    # Ineq dual variable
    s = con_data.ineq_duals

    # Feasible computation
    mu = solver_data.perturbation
    S = Diagonal(s[t])
    r = S * constraint_evaluations[t] .+ mu
    cinv = 1.0 ./ constraint_evaluations[t]
    SCinv = Diagonal(s[t] .* cinv)

    # Iteratively bump regularisation
    code = 0
    reg = options.start_reg
    
    # for reg=options.start_reg:options.reg_step:options.end_reg
    while reg < options.end_reg
        policy.uu_tmp[t] .= Quu[t] + quu[t] * (1.6^reg - 1) .- Qsu[t]' * SCinv * Qsu[t]
        (_, code) = LAPACK.potrf!('U', policy.uu_tmp[t])
        if code > 0
            reg += options.reg_step
            continue
        end
        kK[t] .= -hcat(Qu[t] - Qsu[t]' * (cinv .* r), Qux[t] - Qsu[t]' * SCinv * Qsx[t])
        LAPACK.potrs!('U', policy.uu_tmp[t], kK[t])
        break
    end
    if code > 0  # cycled through all regularization values and still not PD
        error("Regularisation failed for all values")
    end
    
    options.reg = reg

    # Update gains
    ku[t] = kK[t][:, 1]
    Ku[t] = kK[t][:, 2:end]
    ks[t] = -cinv .* (r .+ S * Qsu[t] * ku[t])
    Ks[t] = -SCinv * (Qsx[t] .+ Qsu[t] * Ku[t])
    
    # Update value function approximation
    Quu[t] .-= Qsu[t]' * SCinv * Qsu[t]
    Qux[t] .-= transpose(Qsx[t]' * SCinv * Qsu[t])
    Qxx[t] .-= Qsx[t]' * SCinv * Qsx[t]
    Qu[t] .-= Qsu[t]' * (cinv .* r)
    Qx[t] .-= Qsx[t]' * (cinv .* r)

    # Return r to calculate the mu-err later 
    return r
end

function compute_gains_and_update_infeasible!(policy, problem, solver_data, options, t)

    # Constraint Data 
    con_data = problem.objective.costs.constraint_data
    # Gains 
    Ku = policy.Ku
    ku = policy.ku
    Ks = policy.Ks
    ks = policy.ks
    Ky = policy.Ky
    ky = policy.ky
    kK = policy.kK_tmp
    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Constraint jacobians
    Qsx = con_data.jacobian_state
    Qsu = con_data.jacobian_action
    # Cost hessian 
    quu = problem.objective.hessian_action_action
    # Constraint evaluation
    constraint_evaluations = con_data.inequalities
    # Ineq dual variable
    s = con_data.ineq_duals
    # Slack 
    y = con_data.slacks

    # Infeasible computation
    mu = solver_data.perturbation
    S = Diagonal(s[t])
    r = s[t] .* y[t] .- mu
    rhat = s[t] .* (constraint_evaluations[t] .+ y[t]) .- r
    yinv = 1.0 ./ y[t]
    SYinv = Diagonal(s[t] .* yinv)

    # Iteratively bump regularisation
    code = 0
    reg = options.start_reg
    
    # for reg=options.start_reg:options.reg_step:options.end_reg
    while reg < options.end_reg
        policy.uu_tmp[t] .= Quu[t] + quu[t] * (1.6^reg - 1) .+ Qsu[t]' * SYinv * Qsu[t]
        policy.uu_tmp[t] .+= Quu[t]
        (_, code) = LAPACK.potrf!('U', policy.uu_tmp[t])
        if code > 0
            reg += options.reg_step
            continue
        end
        kK[t] .= -hcat(Qu[t] + Qsu[t]' * (yinv .* rhat), Qux[t] + Qsu[t]' * SYinv * Qsx[t])
        LAPACK.potrs!('U', policy.uu_tmp[t], kK[t])
        break
    end
    if code > 0  # cycled through all regularization values and still not PD
        error("Regularisation failed for all values")
    end
    
    options.reg = reg
    
    # Update gains
    ku[t] = kK[t][:, 1]
    Ku[t] = kK[t][:, 2:end]
    ks[t] = yinv .* (rhat .+ S * Qsu[t] * ku[t])
    Ks[t] = SYinv * (Qsx[t] + Qsu[t] * Ku[t])
    ky[t] = -(constraint_evaluations[t] .+ y[t]) - Qsu[t] * ku[t]
    Ky[t] = -Qsx[t] - Qsu[t] * Ku[t]

    # Update value function approximation 
    Quu[t] .+= Qsu[t]' * SYinv * Qsu[t]
    Qux[t] .+= transpose(Qsx[t]' * SYinv * Qsu[t])
    Qxx[t] .+= Qsx[t]' * SYinv * Qsx[t]
    Qu[t] .+= Qsu[t]' * (yinv .* rhat)
    Qx[t] .+= Qsx[t]' * (yinv .* rhat)
    # Return r to calculate the mu-err later 
    return r

end

function update_value_function!(policy, t)
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