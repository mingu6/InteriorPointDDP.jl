function compute_gains_and_update_feasible!(policy, problem, solver_data, options, t)

    # Constraint Data 
    con_data = problem.objective.costs.constraint_data
    # Gains 
    Ku = policy.Ku
    ku = policy.ku
    Ks = policy.Ks
    ks = policy.ks
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
    R = nothing
    reg = options.start_reg
    while reg <= options.end_reg
        if reg >= options.end_reg && isnothing(R)
            error("Regularisation failed for all values")
        end
        Quu_reg = Quu[t] + quu[t] * (1.6^reg - 1) 
        res = Quu_reg .- Qsu[t]' * SCinv * Qsu[t]
        chol = cholesky(res, check=false)
        if issuccess(chol)
            R = chol.U
            break
        else
            reg = reg + options.reg_step
        end 
    end
    options.reg = reg

    Qxu = transpose(Qux[t])
    b = hcat(Qu[t] - Qsu[t]' * (cinv .* r), Qxu' - Qsu[t]' * SCinv * Qsx[t])
    kK = -R \ (R' \ b)

    # Update gains
    ku[t] = kK[:, 1]
    Ku[t] = kK[:, 2:end]
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
    R = nothing
    reg = options.start_reg
    while reg <= options.end_reg + 1
        if reg >= options.end_reg + 1 && isnothing(R)
            error("Regularisation failed for all values")
        end
        Quu_reg = Quu[t] + quu[t] * (1.6^reg - 1) 
        res = Quu_reg .+ Qsu[t]' * SYinv * Qsu[t]
        chol = cholesky(res, check=false)
        if issuccess(chol)
            R = chol.U
            break
        else
            reg = reg + options.reg_step
        end 
    end

    Qxu = transpose(Qux[t])
    b = hcat(Qu[t] + Qsu[t]' * (yinv .* rhat), Qxu' + Qsu[t]' * SYinv * Qsx[t])
    kK = -R \ (R' \ b)

    # Update gains
    ku[t] = kK[:, 1]
    Ku[t] = kK[:, 2:end]
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

    # Vx[t] .=  Qx[t] + (K[t]' * Quu[t] * k[t]) + (K[t]' * Qu[t]) + (Qux[t]' * k[t])
    mul!(policy.u_tmp[t], Quu[t], ku[t])
    mul!(Vx[t], transpose(Ku[t]), policy.u_tmp[t])
    mul!(Vx[t], transpose(Ku[t]), Qu[t], 1.0, 1.0) # apply appropriate scaling 
    mul!(Vx[t], transpose(Qux[t]), ku[t], 1.0, 1.0) # apply appropriate scaling 
    Vx[t] .+= Qx[t]
end