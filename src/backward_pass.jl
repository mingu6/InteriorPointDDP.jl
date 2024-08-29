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
    # Bound constraints
    bounds = problem.bounds
    
    x, u, h = primal_trajectories(problem, mode=mode)
    ϕ, vl, vu = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = 0.

    reg = 0.0
    
    while reg <= options.reg_max
        data.status = 0
        Vxx[N] .= lxx[N]
        Vx[N] .= lx[N]
        
        for k = N-1:-1:1
            num_action = length(u[k])

            bk = bounds[k]
            il = u[k][bk.indices_lower] - bk.lower[bk.indices_lower]
            iu = bk.upper[bk.indices_upper] - u[k][bk.indices_upper]
            σl = vl[k][bk.indices_lower] ./ il
            σu = vu[k][bk.indices_upper] ./ iu

            # Qx = lx + fx' * Vx
            Qx[k] .= lx[k]
            mul!(Qx[k], transpose(fx[k]), Vx[k+1], 1.0, 1.0)

            # Qu = lu + fu' * Vx + μ log(u - ul) + μ log(uu - u)
            Qu[k] .= lu[k]
            mul!(Qu[k], transpose(fu[k]), Vx[k+1], 1.0, 1.0)
            Qu[k][bk.indices_lower] .-= μ ./ il
            Qu[k][bk.indices_upper] .+= μ ./ iu
            
            # Qxx = lxx + fx' * Vxx * fx
            mul!(policy.xx_tmp[k], transpose(fx[k]), Vxx[k+1])
            mul!(Qxx[k], policy.xx_tmp[k], fx[k])
            Qxx[k] .+= lxx[k]
    
            # Quu = luu + fu' * Vxx * fu + Σ
            mul!(policy.ux_tmp[k], transpose(fu[k]), Vxx[k+1])
            mul!(Quu[k], policy.ux_tmp[k], fu[k])
            Quu[k] .+= luu[k]
            Quu[k][diagind(Quu[k])[bk.indices_lower]] .+= σl
            Quu[k][diagind(Quu[k])[bk.indices_upper]] .+= σu
    
            # Qux = lux + fu' * Vxx * fx
            mul!(policy.ux_tmp[k], transpose(fu[k]), Vxx[k+1])
            mul!(Qux[k], policy.ux_tmp[k], fx[k])
            Qux[k] .+= lux[k]
            
            # apply second order terms to Q for full DDP, i.e., Vx * fxx, Vx * fuu, Vx * fxu
            if !options.quasi_newton
                hessian_vector_prod!(fxx[k], fux[k], fuu[k], problem.model.dynamics[k], x[k], u[k], Vx[k+1])
                Qxx[k] .+= fxx[k]
                Qux[k] .+= fux[k]
                Quu[k] .+= fuu[k]
                hessian_vector_prod!(hxx[k], hux[k], huu[k], constr_data.constraints[k], x[k], u[k], ϕ[k])
            end
            # setup linear system in backward pass
            policy.lhs_tl[k] .= Quu[k]
            policy.lhs_tr[k] .= transpose(hu[k])
            fill!(policy.lhs_br[k], 0.0)
            if !options.quasi_newton
                policy.lhs_tl[k] .+= huu[k]
            end
            policy.lhs[k] .= Symmetric(policy.lhs[k])

            policy.rhs_t[k] .= -Qu[k]
            mul!(policy.rhs_t[k], transpose(hu[k]), ϕ[k], -1.0, 1.0)
            policy.rhs_b[k] .= -h[k]

            policy.rhs_x_t[k] .= -Qux[k]
            policy.rhs_x_b[k] .= -hx[k]
            if !options.quasi_newton
                policy.rhs_x_t[k] .-= hux[k]
            end

            # inertia calculation and correction
            policy.lhs_tl[k][diagind(policy.lhs_tl[k])] .+= reg
            policy.lhs_br[k][diagind(policy.lhs_br[k])] .-= δ_c

            policy.lhs_bk[k], data.status, reg, δ_c = inertia_correction!(policy.lhs[k], num_action,
                        μ, reg, data.reg_last, options; rook=true)

            data.status != 0 && break

            kuϕ[k] .= policy.lhs_bk[k] \ policy.rhs[k]
            Kuϕ[k] .= policy.lhs_bk[k] \ policy.rhs_x[k]

            # update gains for ineq. dual variables TODO: merge when gains are merged
            kvl[k][bk.indices_lower] .= μ ./ il - vl[k][bk.indices_lower] - σl .* ku[k][bk.indices_lower]
            kvu[k][bk.indices_upper] .= μ ./ iu - vu[k][bk.indices_upper] + σu .* ku[k][bk.indices_upper]
            Kvl[k][bk.indices_lower, :] .= -σl .* Ku[k][bk.indices_lower, :]
            Kvu[k][bk.indices_upper, :] .= σu .* Ku[k][bk.indices_upper, :]

            # Update return function approx. for next timestep 
            # Vxx = Q̂xx + Q̂ux' * Ku + Ku * Q̂ux' + Ku' Q̂uu' * Ku
            mul!(policy.ux_tmp[k], Quu[k], Ku[k])
            mul!(Vxx[k], transpose(Ku[k]), Qux[k])

            mul!(Vxx[k], transpose(Qux[k]), Ku[k], 1.0, 1.0)
            Vxx[k] .+= Qxx[k]
            mul!(Vxx[k], transpose(Ku[k]), policy.ux_tmp[k], 1.0, 1.0)
            Vxx[k] .= Symmetric(Vxx[k])

            # Vx = Q̂x + Ku' * Q̂u + [Q̂uu Ku + Q̂ux]^T ku
            policy.ux_tmp[k] .+= Qux[k]
            mul!(Vx[k], transpose(policy.ux_tmp[k]), ku[k])
            mul!(Vx[k], transpose(Ku[k]), Qu[k], 1.0, 1.0)
            Vx[k] .+= Qx[k]
        end
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find positive definite iteration matrix."))
end

# TODO: transpose allocates and slows things down
# TODO: lots of unnecessary tranposes? fx, fu, ux_tmp etc
