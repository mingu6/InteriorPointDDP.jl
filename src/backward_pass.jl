using LinearAlgebra

function backward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; mode=:nominal, verbose::Bool=false)
    H = length(problem.states)
    constr_data = problem.constraints
    ϕ::Float64 = 0.0
    code = 0

    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    fxx = problem.model.hessian_prod_state_state
    fux = problem.model.hessian_prod_action_state
    fuu = problem.model.hessian_prod_action_action
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
    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    
    x, u, _ = primal_trajectories(problem, mode=mode)
    c, s, y = dual_trajectories(constr_data, mode=mode)
    
    # See (9) ad (17) in Pavlov et al. Same except swap C_t for -Y_t
    cy = options.feasible ? c : y
    α = options.feasible ? -1.0 : 1.0
    
    while ϕ <= options.ϕ_max
        data.status = true
        Vxx[H] .= qxx[H]
        Vx[H] .= qx[H]
        
        for t = H-1:-1:1
            # lx = qx + Qsx' * s
            mul!(Qx[t], transpose(Qsx[t]),s[t])
            Qx[t] .+= qx[t]
            # Qx = lx + fx' * Vx
            mul!(Qx[t], transpose(fx[t]), Vx[t+1], 1.0, 1.0)

            # lu = qu + Qsu' * s
            mul!(Qu[t], transpose(Qsu[t]), s[t])
            Qu[t] .+= qu[t]
            # Qu = lu + fu' * Vx
            mul!(Qu[t], transpose(fu[t]), Vx[t+1], 1.0, 1.0)
            
            # Qxx = qxx + fx' * Vxx * fx
            if options.reg_state
                Vxx[t+1][diagind(Vxx[t+1])] .+= ϕ
            end
            mul!(policy.xx̂_tmp[t], transpose(fx[t]), Vxx[t+1])
            mul!(Qxx[t], policy.xx̂_tmp[t], fx[t])
            Qxx[t] .+= qxx[t]
    
            # Quu = quu + fu' * Vxx * fu
            mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Quu[t], policy.ux̂_tmp[t], fu[t])
            Quu[t] .+= quu[t]
    
            # Qux = qux + fu' * Vxx * fx
            mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Qux[t], policy.ux̂_tmp[t], fx[t])
            Qux[t] .+= qux[t]
            
            # apply second order terms to Q for full DDP, i.e., Vx * fxx, Vx * fuu, Vx * fxu
            if !options.quasi_newton
                hessian_vector_prod!(fxx[t], fux[t], fuu[t], problem.model.dynamics[t], x[t], u[t], problem.parameters[t], Vx[t+1])
                Qxx[t] .+= fxx[t]
                Qux[t] .+= fux[t]
                Quu[t] .+= fuu[t]
                
                hessian_vector_prod!(fxx[t], fux[t], fuu[t], constr_data.constraints[t], x[t], u[t], problem.parameters[t], s[t])
                Qxx[t] .+= fxx[t]
                Qux[t] .+= fux[t]
                Quu[t] .+= fuu[t]
            end
            
            # S * r for infeasible and feasible, see for example RHS of (9)
            fill!(policy.s_tmp[t], data.μ)
            policy.s_tmp[t] ./= s[t]
            policy.s_tmp[t] .+= c[t]
            
            # cache S * C^-1 * Qsu (or S * Y^-1 * Qsu)
            policy.su_tmp[t] .= Qsu[t]
            policy.su_tmp[t] .*= s[t]
            policy.su_tmp[t] ./= cy[t]
            
            # cache S * C^-1 Qsx
            policy.sx_tmp[t] .= Qsx[t]
            policy.sx_tmp[t] .*= s[t]
            policy.sx_tmp[t] ./= cy[t]
            
            # apply Schur complement over LHS of bottom right block of (9) or reduced system in (17) in Pavlov et al. Setup block inverse.
            
            # Q̂uu = Quu - Qsu' * S * C^-1 * Qsu (or Q̂uu = Quu + Qsu' * Y * C^-1 * Qsu)
            mul!(Quu[t], transpose(Qsu[t]), policy.su_tmp[t], α, 1.0)
            # Q̂ux = Qux - Qsu' * S * C^-1 * Qsx (or Q̂ux = Qux + Qsx' * Y * C^-1 * Qsx)
            mul!(Qux[t], transpose(policy.su_tmp[t]), Qsx[t], α, 1.0)
            # Q̂u = Qu - Qsu' * S * C^-1 * (S * r) (or Q̂u = Qu + Qsu' * S * Y^-1 (S * r))
            mul!(Qu[t], transpose(policy.su_tmp[t]), policy.s_tmp[t], α, 1.0)
            # Q̂xx = Qxx - Qsx' * S * C^-1 * Qsx (or Q̂xx = Qxx + Qsx' * S * Y^-1 * Qsx)
            mul!(Qxx[t], transpose(Qsx[t]), policy.sx_tmp[t], α, 1.0)
            # Q̂x = Qx - Qsx' * S * C^-1 * (S * r) (or Q̂x = Qx + Qsx' * S * Y^-1 * (S * r))
            mul!(Qx[t], transpose(policy.sx_tmp[t]), policy.s_tmp[t], α, 1.0)
            
            # factorize Schur complement and apply regularisation, i.e., Q̂uu[t] + I_m * ϕ if required
            policy.uu_tmp[t] .= Quu[t]
            if !options.reg_state
                policy.uu_tmp[t][diagind(policy.uu_tmp[t])] .+= ϕ
            end
            (_, code) = LAPACK.potrf!('U', policy.uu_tmp[t])
            if code != 0
                if ϕ == 0.0  # not initialised
                    ϕ = (data.ϕ_last == 0.0) ? options.ϕ_1 : max(options.ϕ_min, options.ψ_m * data.ϕ_last)
                else
                    ϕ = (data.ϕ_last == 0.0) ? ϕ * options.ψ_p_1 : ϕ * options.ψ_p_2
                end
                data.status = false
                break
            end
            # see block inverse formula https://en.wikipedia.org/wiki/Block_matrix (D and complement of D is invertible case)
            ku[t] .= -Qu[t]
            Ku[t] .= -Qux[t]
            LAPACK.potrs!('U', policy.uu_tmp[t], ku[t])
            LAPACK.potrs!('U', policy.uu_tmp[t], Ku[t])
            
            # ks[t] .= diag(c[t] or y[t])^-1 .* (r .+ s[t] .* Qsu[t] * ku[t])
            # note: δs = -S * C^-1 * (S * r) - (S * C^-1 * Qsu) * ku - S * C-1 * (sx + Qsu * Ku) δx (or replace C^-1 for -Y^1)
            
            # ks = -S * C ^-1 * (S * r) - (S * C^-1 * Qsu) * ku (or S * Y^-1 * (S * r) + (S * Y^-1 * Qsu) * ku)
            ks[t] .= policy.s_tmp[t]
            ks[t] .*= s[t]
            ks[t] ./= cy[t]
            mul!(ks[t], policy.su_tmp[t], ku[t], α, α)
            
            # Ks[t] .= s_cy_inv .* (Qsx[t] + Qsu[t] * Ku[t])
            # Ks = -S * C^-1 * (Qsx + Qsu * Ku) (or Ks = S * Y^-1 * (Qsx + Qsu * Ku))
            Ks[t] .= policy.sx_tmp[t]
            mul!(Ks[t], policy.su_tmp[t], Ku[t], α, α)
            
            if !options.feasible
                # ky = -(c + y) - Qsu * ku
                ky[t] .= c[t]
                ky[t] .+= y[t]
                mul!(ky[t], Qsu[t], ku[t], -1.0, -1.0)
                
                # Ky = -Qsx - Qsu * Ku
                Ky[t] .= Qsx[t]
                mul!(Ky[t], Qsu[t], Ku[t], -1.0, -1.0)
            end
            
            # Update return function approx. for next timestep 
            # Vxx = Q̂xx + Q̂ux' * Ku + Ku * Q̂ux' + Ku' Q̂ux' * Ku
            mul!(policy.ux_tmp[t], Quu[t], Ku[t])
            
            mul!(Vxx[t], transpose(Ku[t]), Qux[t])
            Vxx[t] .+= transpose(Vxx[t])
            Vxx[t] .+= Qxx[t]
            mul!(Vxx[t], transpose(Ku[t]), policy.ux_tmp[t], 1.0, 1.0)
        
            # Vx = Q̂x + Ku' * Q̂u + [Q̂uu Ku + Q̂ux]^T ku
            policy.ux_tmp[t] .+= Qux[t]
            mul!(Vx[t], transpose(policy.ux_tmp[t]), ku[t])
            mul!(Vx[t], transpose(Ku[t]), Qu[t], 1.0, 1.0)
            Vx[t] .+= Qx[t]
        end
        data.status && break
    end
    data.ϕ_last = ϕ
    !data.status && (verbose && (@warn "Backward pass failure, unable to find positive definite iteration matrix."))
end

