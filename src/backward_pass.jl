using LinearAlgebra

function backward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; mode=:nominal, verbose::Bool=false)
    N = problem.horizon
    constr_data = problem.constr_data
    reg::Float64 = 0.0

    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    fxx = problem.model.hessian_prod_state_state
    fux = problem.model.hessian_prod_action_state
    fuu = problem.model.hessian_prod_action_action
    # Jacobian constr_data 
    hx = constr_data.jacobian_state
    hu = constr_data.jacobian_action
    # Second order cosntraint stuff
    hxx = constr_data.hessian_prod_state_state
    hux = constr_data.hessian_prod_action_state
    huu = constr_data.hessian_prod_action_action
    # Cost gradients
    lx = problem.cost_data.gradient_state
    lu = problem.cost_data.gradient_action
    # Cost hessians
    lxx = problem.cost_data.hessian_state_state
    luu = problem.cost_data.hessian_action_action
    lux = problem.cost_data.hessian_action_state
    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Feedback gains (linear feedback policy)
    Kuϕ = policy.gains_main.Kuϕ
    kuϕ = policy.gains_main.kuϕ
    Ku = policy.gains_main.Ku
    ku = policy.gains_main.ku
    kvl = policy.gains_main.kvl
    kvu = policy.gains_main.kvu
    Kvl = policy.gains_main.Kvl
    Kvu = policy.gains_main.Kvu
    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    
    x, u, h, il, iu = primal_trajectories(problem, mode=mode)
    ϕ, vl, vu = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = 0.
    
    while reg <= options.reg_max
        data.status = true
        Vxx[N] .= lxx[N]
        Vx[N] .= lx[N]
        
        for t = N-1:-1:1
            num_actions = length(lu[t])

            # Qx = lx + fx' * Vx
            Qx[t] .= lx[t]
            mul!(Qx[t], transpose(fx[t]), Vx[t+1], 1.0, 1.0)

            # Qu = lu + fu' * Vx
            Qu[t] .= lu[t]
            mul!(Qu[t], transpose(fu[t]), Vx[t+1], 1.0, 1.0)
            add_barrier_grad!(Qu[t], il[t], iu[t], μ)
            
            # Qxx = lxx + fx' * Vxx * fx
            mul!(policy.xx_tmp[t], transpose(fx[t]), Vxx[t+1])
            mul!(Qxx[t], policy.xx_tmp[t], fx[t])
            Qxx[t] .+= lxx[t]
    
            # Quu = luu + fu' * Vxx * fu
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Quu[t], policy.ux_tmp[t], fu[t])
            Quu[t] .+= luu[t]
            add_primal_dual!(Quu[t], il[t], iu[t], vl[t], vu[t])
    
            # Qux = lux + fu' * Vxx * fx
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Qux[t], policy.ux_tmp[t], fx[t])
            Qux[t] .+= lux[t]
            
            # apply second order terms to Q for full DDP, i.e., Vx * fxx, Vx * fuu, Vx * fxu
            if !options.quasi_newton
                hessian_vector_prod!(fxx[t], fux[t], fuu[t], problem.model.dynamics[t], x[t], u[t], Vx[t+1])
                Qxx[t] .+= fxx[t]
                Qux[t] .+= fux[t]
                Quu[t] .+= fuu[t]
                hessian_vector_prod!(hxx[t], hux[t], huu[t], constr_data.constraints[t], x[t], u[t], ϕ[t])
            end
            # setup linear system in backward pass
            policy.lhs_tl[t] .= Quu[t]
            policy.lhs_tr[t] .= transpose(hu[t])
            policy.lhs_bl[t] .= hu[t]
            fill!(policy.lhs_br[t], 0.0)
            if !options.quasi_newton
                policy.lhs_tl[t] .+= huu[t]
            end

            policy.rhs_t[t] .= -Qu[t]
            mul!(policy.rhs_t[t], transpose(hu[t]), ϕ[t], -1.0, 1.0)
            policy.rhs_b[t] .= -h[t]

            policy.rhs_x_t[t] .= -Qux[t]
            policy.rhs_x_b[t] .= -hx[t]
            if !options.quasi_newton
                policy.rhs_x_t[t] .-= hux[t]
            end

            # inertia calculation and correction
            policy.lhs_tl[t][diagind(policy.lhs_tl[t])] .+= reg
            policy.lhs_br[t][diagind(policy.lhs_br[t])] .-= δ_c

            policy.lhs_bk[t], data.status, reg, δ_c = inertia_correction!(policy.lhs[t], num_actions,
                        μ, reg, data.reg_last, options; rook=true)
            !data.status && break

            kuϕ[t] .= policy.lhs_bk[t] \ policy.rhs[t]
            Kuϕ[t] .= policy.lhs_bk[t] \ policy.rhs_x[t]

            # update gains for ineq. duals
            gains_ineq!(kvl[t], kvu[t], Kvl[t], Kvu[t], il[t], iu[t], vl[t], vu[t], ku[t], Ku[t], μ)

            # Update return function approx. for next timestep 
            # Vxx = Q̂xx + Q̂ux' * Ku + Ku * Q̂ux' + Ku' Q̂uu' * Ku
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
    data.reg_last = reg
    !data.status && (verbose && (@warn "Backward pass failure, unable to find positive definite iteration matrix."))
end

function add_primal_dual!(Quu, ineq_lower, ineq_upper, dual_lower, dual_upper)
    m = length(ineq_lower)
    for i = 1:m
        if !isinf(ineq_lower[i])
            Quu[i, i] += dual_lower[i] / ineq_lower[i]
        end
        if !isinf(ineq_upper[i])
            Quu[i, i] += dual_upper[i] / ineq_upper[i]
        end
    end
end

function add_barrier_grad!(Qu, ineq_lower, ineq_upper, μ)
    m = length(ineq_lower)
    for i = 1:m
        if !isinf(ineq_lower[i])
            Qu[i] -= μ / ineq_lower[i]
        end
        if !isinf(ineq_upper[i])
            Qu[i] += μ / ineq_upper[i]
        end
    end
end

function gains_ineq!(kvl, kvu, Kvl, Kvu, il, iu, vl, vu, ku, Ku, μ)
    m = length(il)
    for i = 1:m
        σ_l = vl[i] / il[i]
        σ_u = vu[i] / iu[i]
        kvl[i] = isinf(il[i]) ? 0.0 : μ / il[i] - vl[i] - σ_l * ku[i]
        kvu[i] = isinf(iu[i]) ? 0.0 : μ / iu[i] - vu[i] + σ_u * ku[i]
        Kvl[i, :] .= isinf(il[i]) ? 0.0 : -σ_l * Ku[i, :]
        Kvu[i, :] .= isinf(iu[i]) ? 0.0 : σ_u * Ku[i, :]
    end
end
