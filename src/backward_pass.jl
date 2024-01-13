using LinearAlgebra

function backward_pass!(policy::PolicyData, problem::ProblemData, solver_data::SolverData,
    options::Options, mode=:nominal)

    N = length(problem.states)
    constraints = problem.objective.costs.constraint_data
    reg::Float64 = options.start_reg

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
    # Feedback gains (linear feedback policy)
    Ku = policy.Ku
    ku = policy.ku
    Ks = policy.Ks
    ks = policy.ks
    Ky = policy.Ky
    ky = policy.ky
    kkt_soln = policy.kkt_soln_tmp
    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    
    c = constraints.inequalities
    s = constraints.ineq_duals
    y = constraints.slacks

    # terminal value function
    # TODO: Extension: Implement terminal constraints 
    Vxx[N] .= qxx[N]
    Vx[N] .= qx[N]

    for t = N-1:-1:1
        # update Q function approx.
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
        
        cinv = 1.0 ./ c[t]
        
        code = 0
        if options.feasible
            r = s[t] .* c[t] .+ solver_data.μ_j
            s_cinv = s[t] .* cinv
            # update local feedback policy/gains
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
            ku[t] = kkt_soln[t][:, 1]
            Ku[t] = kkt_soln[t][:, 2:end]
            ks[t] = -cinv .* (r .+ s[t] .* Qsu[t] * ku[t])
            Ks[t] = -s_cinv .* (Qsx[t] .+ Qsu[t] * Ku[t])
            
            # update Q to \hat{Q} as in (12) in Pavlov et al.
            Quu[t] .-= Qsu[t]' * (s_cinv .* Qsu[t])
            Qux[t] .-= transpose(Qsx[t]' * (s_cinv .* Qsu[t]))
            Qxx[t] .-= Qsx[t]' * (s_cinv .* Qsx[t])
            Qu[t] .-= Qsu[t]' * (cinv .* r)
            Qx[t] .-= Qsx[t]' * (cinv .* r)
        else
            # update local feedback policy/gains
            r = s[t] .* y[t] .- solver_data.μ_j
            rhat = s[t] .* (c[t] .+ y[t]) .- r
            yinv = 1.0 ./ y[t]
            s_yinv = s[t] .* yinv
        
            # Iteratively bump regularisation
            code = 0
            reg = options.start_reg
            
            # for reg=options.start_reg:options.reg_step:options.end_reg
            while reg < options.end_reg
                policy.uu_tmp[t] .= Quu[t] + quu[t] * (1.6^reg - 1) .+ Qsu[t]' * (s_yinv .* Qsu[t])
                (_, code) = LAPACK.potrf!('U', policy.uu_tmp[t])
                if code > 0
                    reg += options.reg_step
                    continue
                end
                kkt_soln[t] .= -hcat(Qu[t] + Qsu[t]' * (yinv .* rhat), Qux[t] + Qsu[t]' * (s_yinv .* Qsx[t]))
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
            ks[t] = yinv .* (rhat .+ s[t] .* Qsu[t] * ku[t])
            Ks[t] = s_yinv .* (Qsx[t] + Qsu[t] * Ku[t])
            ky[t] = -(c[t] .+ y[t]) - Qsu[t] * ku[t]
            Ky[t] = -Qsx[t] - Qsu[t] * Ku[t]
        
            # Update value function approximation 
            Quu[t] .+= Qsu[t]' * (s_yinv .* Qsu[t])
            Qux[t] .+= transpose(Qsx[t]' * (s_yinv .* Qsu[t]))
            Qxx[t] .+= Qsx[t]' * (s_yinv .* Qsx[t])
            Qu[t] .+= Qsu[t]' * (yinv .* rhat)
            Qx[t] .+= Qsx[t]' * (yinv .* rhat)
        end
        
        # Update value function approx.
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
end
