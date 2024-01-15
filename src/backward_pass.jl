using LinearAlgebra

function backward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options, mode=:nominal)
    N = length(problem.states)
    constr_data = problem.constraints
    reg::Float64 = options.start_reg

    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    # fxx = problem.model.hessian_state_state
    # fxu = problem.model.hessian_state_action
    # fuu = problem.model.hessian_action_action
    # Jacobian constr_data 
    Qsx = constr_data.jacobian_state
    Qsu = constr_data.jacobian_action
    # Cost gradients
    qx = problem.costs.gradient_state
    qu = problem.costs.gradient_action
    # Cost hessians
    qxx = problem.costs.hessian_state_state
    quu = problem.costs.hessian_action_action
    qux = problem.costs.hessian_action_state
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
    
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks

    # terminal value function
    # TODO: Extension: Implement terminal constr_data 
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
        
        cy = options.feasible ? -c[t] : y[t]
        # for infeasible, directly computes r̂ from (17) is the same as r in (9) in Pavlov et al.
        policy.r_tmp[t] .= s[t]
        policy.r_tmp[t] .*= c[t]
        policy.r_tmp[t] .+= data.μ_j
        policy.r_tmp[t] ./= cy  # r̂ ./ c[t] or r̂ ./ y[t]
        
        policy.su_tmp[t] .= Qsu[t]
        policy.su_tmp[t] .*= s[t]
        policy.su_tmp[t] ./= cy
        
        policy.sx_tmp[t] .= Qsx[t]
        policy.sx_tmp[t] .*= s[t]
        policy.sx_tmp[t] ./= cy
        
        # r = s[t] .* c[t] .+ data.μ_j
        
        # update local feedback policy/gains, e.g., LHS of (11)
        code = 0
        while reg < options.end_reg
            # policy.uu_tmp[t] .= Quu[t] + quu[t] * (1.6^reg - 1) .+ Qsu[t]' * diag(s[t]) * diag(c[t] or y[t])^-1 .* Qsu[t]
            policy.uu_tmp[t] .= quu[t]
            policy.uu_tmp[t] .*= (1.6^reg - 1.)
            policy.uu_tmp[t] .+= Quu[t]
            mul!(policy.uu_tmp[t], transpose(Qsu[t]), policy.su_tmp[t], 1.0, 1.0)
            (_, code) = LAPACK.potrf!('U', policy.uu_tmp[t])
            if code > 0
                reg += options.reg_step
                continue
            end
            # kkt_soln[t] .= -hcat(Qu[t] + Qsu[t]' * diag(c[t] or y[t])^-1 * r, Qux[t] + Qsu[t]' * diag(s[t]) * diag(c[t] or y[t])^-1 .* Qsx[t])
            kkt_soln[t][:, 1] .= Qu[t]
            mul!(@view(kkt_soln[t][:, 1]), transpose(Qsu[t]), policy.r_tmp[t], -1.0, -1.0)
            kkt_soln[t][:, 2:end] .= Qux[t]
            mul!(@view(kkt_soln[t][:, 2:end]), transpose(Qsu[t]), policy.sx_tmp[t], -1.0, -1.0)
            LAPACK.potrs!('U', policy.uu_tmp[t], kkt_soln[t])
            break
        end
        if code > 0  # cycled through all regularization values and still not PD
            error("Regularisation failed for all values")
        end
        options.reg = reg
        
        ku[t] .= kkt_soln[t][:, 1]
        Ku[t] .= kkt_soln[t][:, 2:end]
        
        # ks[t] .= diag(c[t] or y[t])^-1 .* (r .+ s[t] .* Qsu[t] * ku[t])
        mul!(ks[t], policy.su_tmp[t], ku[t])
        ks[t] .+= policy.r_tmp[t]
        
        # Ks[t] .= s_cy_inv .* (Qsx[t] + Qsu[t] * Ku[t])
        mul!(Ks[t], policy.su_tmp[t], Ku[t])
        Ks[t] .+= policy.sx_tmp[t]
        
        if !options.feasible
            # ky[t] .= -(c[t] .+ y[t]) - Qsu[t] * ku[t]
            ky[t] .= c[t]
            ky[t] .+= y[t]
            mul!(ky[t], Qsu[t], ku[t], -1.0, -1.0)
            
            # Ky[t] .= -Qsx[t] - Qsu[t] * Ku[t]
            Ky[t] .= Qsx[t]
            mul!(Ky[t], Qsu[t], Ku[t], -1.0, -1.0)
        end
    
        # Update value function approximation
        # Quu[t] = Quu[t] + Qsu[t]' * diag(s[t]) * diag(c[t] or y[t])^-1 * Qsu[t]
        mul!(Quu[t], transpose(Qsu[t]), policy.su_tmp[t], 1.0, 1.0)
        # Qux[t] = Qux[t] + Qsx[t]' * diag(s[t]) * diag(c[t] or y[t])^-1 * Qsu[t]
        mul!(Qux[t], transpose(policy.su_tmp[t]), Qsx[t], 1.0, 1.0)
        # Qxx[t] = Qsx[t]' + diag(s[t]) * diag(c[t] or y[t])^-1 * Qsx[t]
        mul!(Qxx[t], transpose(Qsx[t]), policy.sx_tmp[t], 1.0, 1.0)
        # Qu[t] .+= Qsu[t]' * diag(c[t] or y[t])^-1 * r
        mul!(Qu[t], transpose(Qsu[t]), policy.r_tmp[t], 1.0, 1.0)
        # Qx[t] .+= Qsx[t]' * diag(c[t] or y[t])^-1 * r
        mul!(Qx[t], transpose(Qsx[t]), policy.r_tmp[t], 1.0, 1.0)
        
        # Update value function approx.
        # Vxx[t] .=  Qxx[t] + (K[t]' * (Quu[t] * K[t])) + (K[t]' * Qux[t]) + (Qux[t]' * K[t])
        mul!(policy.ux_tmp[t], Quu[t], Ku[t])
        mul!(Vxx[t], transpose(Ku[t]), policy.ux_tmp[t])
        mul!(Vxx[t], transpose(Ku[t]), Qux[t], 1.0, 1.0)
        mul!(Vxx[t], transpose(Qux[t]), Ku[t], 1.0, 1.0)
        Vxx[t] .+= Qxx[t]
    
        # Vx[t] .=  Qx[t] + (K[t]' * Quu[t] * k[t]) + (K[t]' * Qu[t]) + (Qux[t]' * k[t])
        mul!(policy.u_tmp[t], Quu[t], ku[t])
        mul!(Vx[t], transpose(Ku[t]), policy.u_tmp[t])
        mul!(Vx[t], transpose(Ku[t]), Qu[t], 1.0, 1.0)
        mul!(Vx[t], transpose(Qux[t]), ku[t], 1.0, 1.0)
        Vx[t] .+= Qx[t]
    end
end
