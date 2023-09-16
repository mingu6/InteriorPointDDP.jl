using LinearAlgebra

function backward_pass!(policy::PolicyData, 
    problem::ProblemData,
    constraint_data::ConstraintData,
    bp::BackwardPassParams,
    fp::ForwardPassParams,
    mode=:nominal, 
    )

    # Horizon 
    H = length(problem.states)

    # Dimensions 
    dim_x = length(problem.states)
    dim_u = length(problem.actions)
    dim_c = length(constraint_data.constraints)

    # Set regularisation
    set_reg!(fp, bp)

    # Errors
    dV = [0.0, 0.0]
    c_err::Float64 = 0
    mu_err::Float64 = 0
    Qu_err::Float64 = 0

    # Current trajectory
    x = problem.states
    u = problem.actions

    # Constraints 
    c = constraint_data.constraints

    # Dual variables 
    # TODO: Double check that these are correct 
    s = problem.dual.ineq
    λ = problem.dual.ineq
    
    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    fxx = problem.model.hessian_state_state
    fxu = problem.model.hessian_state_action
    fuu = problem.model.hessian_action_action
    # Jacobian constraints 
    Qsx = constraint_data.jacobian_state
    Qsu = constraint_data.jacobian_action

    # Cost 
    q = problem.objective.costs
    # Cost gradients
    qx = problem.objective.gradient_state
    qu = problem.objective.gradient_action
    # Cost hessians
    qxx = problem.objective.hessian_state_state
    quu = problem.objective.hessian_action_action
    qux = problem.objective.hessian_action_state

    # Value function approximation
    V = policy.value
    Vx = policy.value.gradient
    Vxx = policy.value.hessian

    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state

    # Gains/Policy
    # TODO: Need one for slack for infeasible ipddp
    # Action-gains (K = Ku) and (k = ku)
    K = policy.K
    k = policy.k

    # Gains for ineq dual variable 
    Ks = policy.Ks
    ks = policy.ks

    # terminal value function
    # TODO: Extension: Implement terminal constraints 
    Vxx[H] .= qxx[H]
    Vx[H] .= qx[H]

    for t = H-1:-1:1
        # Qx[t] .= qx[t] + (Qsx[t]' * s) + (fx[t]' * Vx[t+1])
        mul!(policy.x_tmp, transpose(Qsx[t]), s)
        mul!(Qu[t], transpose(fx[t]), Vx[t+1])
        Qu[t] .+= qx[t] + policy.x_tmp

        # Qu[t] .= qu[t] + (Qsu[t]' * s) + (fu[t]' * Vx[t+1])
        mul!(policy.u_tmp, transpose(Qsu[t]), s)
        mul!(Qu[t], transpose(fu[t]), Vx[t+1])
        Qu[t] .+= qu[t] + policy.u_tmp

        # Qxx[t] .= qxx[t] + ((fx[t]' * Vxx[t+1]) * fx[t])
        mul!(policy.xx̂_tmp[t], transpose(fx[t]), Vxx[t+1])
        mul!(Qxx[t], policy.xx̂_tmp[t], fx[t])
        Qxx[t] .+= qxx[t]

        # Quu[t] .= quu[t] + ((fu[t]' * Vxx[t+1]) * fu[t])
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Quu[t], policy.ux̂_tmp[t], fu[t])
        Quu[t] .+= quu[t]

        # Qux[t] .= qux[t] + ((fu[t]' * Vxx[t+1]) * fx[t])
        # TODO: Is this symmetric to Qxu in the matlab code?
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Qux[t], policy.ux̂_tmp[t], fx[t])
        Qux[t] .+= qux[t]

        S = Diagonal(s)

        Quu_reg = Quu[t] + quu[t] * (1.6^bp.reg - 1) 
        mu = constraint_data.pertubation

        # Here is a bunch of future TODO 
        # 1) Factor out ku and Ku 
        # 2) Optimise cholesky decomp
        if bp.feasible
            r = S * c + mu
            cinv = 1.0 ./ c
            SCinv = Diagonal(s .* cinv)

            cholesky_decomp = cholesky(Quu_reg .- Qsu' * SYinv * Qsu, check=false)
            if !issucess(cholesky_decomp)
                bp.failed = true
                break
            else 
                R = cholesky_decomp.U 
                L = cholesky_decomp.L
                
                kK = -R \ ( R' \ [Qu - Qsu' * (cinv.*r), Qxu' - Qsu' * SCinv * Qsx])
                # Update gains for K (Ku) and k (ku)
                k[t] = kK[:, 1]
                K[t] = kK[:, 2:end]
                # Update gains for ineq dual variable ks and Ks
                ks[t] = -cinv .* (r .+ S * Qsu * ku)
                Ks[t] = -SCinv * (Qsx .+ Qsu * Ku)
                ky = zeros(dim_c)
                Ky = zeros(dim_c, dim_x)

                # TODO: Need to change to work with current timestep - I'm not sure if this is fine right now
                Quu=Quu-Qsu'*SCinv*Qsu;
                Qux=Qux-Qsx'*SCinv*Qsu; # TODO: Assumption that this is symmetric 
                Qxx=Qxx-Qsx'*SCinv*Qsx;

                Qu=Qu-Qsu'*(cinv.*r);
                Qx=Qx-Qsx'*(cinv.*r);
            end
        end

        # TODO: Infeasible computation
        # else 
        #     r = s .* y - mu
        #     rhat = s .* (c + y) - r
        #     yinv = 1.0 ./ y
        #     SYinv = Diagonal(s .* yinv)

        #     cholesky_decomp = cholesky(Quu_reg .+ Qsu' * SYinv * Qsu, check=false)
        #     if !issucess(cholesky_decomp)
        #         bp.failed = true
        #         break
        #     else 
        #         R = cholesky_decomp.U 
        #         kK = -R \ (R' \ [Qu .+ Qsu' * (yinv .* rhat); Qxu' .+ Qsu' * SYinv * Qsx])
        #         k[t] = kK[:, 1]
        #         K[t] = kK[:, 2:end]
        #         ks[t] = yinv .* (rhat .+ S * Qsu * ku)
        #         Ks[t] = SYinv * (Qsx .+ Qsu * Ku)
        #         ky = -(c .+ y) - Qsu * ku
        #         Ky = -Qsx - Qsu * Ku

        #         Quu .+= Qsu' * SYinv * Qsu
        #         Qxu .+= Qsx' * SYinv * Qsu
        #         Qxx .+= Qsx' * SYinv * Qsx
        #         Qu .+= Qsu' * (yinv .* rhat)
        #         Qx .+= Qsx' * (yinv .* rhat)
        #     end

        # Vxx[t] .=  Qxx[t] + (K[t]' * (Quu[t] * K[t])) + (K[t]' * Qux[t]) + (Qux[t]' * K[t])
        mul!(policy.ux_tmp[t], Quu[t], K[t])
        mul!(Vxx[t], transpose(K[t]), policy.ux_tmp[t])
		mul!(Vxx[t], transpose(K[t]), Qux[t], 1.0, 1.0) # apply appropriate scaling 
		mul!(Vxx[t], transpose(Qux[t]), K[t], 1.0, 1.0) # apply appropriate scaling
		Vxx[t] .+= Qxx[t]

        # Vx[t] .=  Qx[t] + (K[t]' * Quu[t] * k[t]) + (K[t]' * Qu[t]) + (Qux[t]' * k[t])
        mul!(policy.u_tmp, Quu[t], k[t])
        mul!(Vx[t], transpose(K[t]), policy.u_tmp)
		mul!(Vx[t], transpose(K[t]), Qu[t], 1.0, 1.0) # apply scaling 
		mul!(Vx[t], transpose(Qux[t]), k[t], 1.0, 1.0) # apply scaling 
		Vx[t] .+= Qx[t]

        # Error mapping 
        Qu_err = max(Qu_err, norm(Qu, Inf))
        mu_err = max(mu_err, norm(r, Inf))
        if !feasible
            c_err = max(c_err, norm(fp.c[:,i] + fp.y[:,i], Inf))
        end
    end

    bp.failed=0;
    bp.opterr=max([Qu_err, c_err, mu_err]);
    
end

function set_reg!(fp::ForwardPassParams, bp::BackwardPassParams)
    if fp.failed || bp.failed
        bp.reg += 1 
    elseif fp.step == 1
        bp.reg -= 1 
    elseif fp.step > 4
        bp.reg += 1
    end

    if bp.reg < 0
        bp.reg = 0 
    elseif bp.reg > 24
        bp.reg = 24
    end
end 
    