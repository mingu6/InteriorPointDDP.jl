using LinearAlgebra

function backward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options; mode=:nominal, verbose::Bool=false)
    N = length(problem.states)
    constr_data = problem.constr_data
    reg::Float64 = 0.0
    code = 0

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
    Kuϕ = policy.Kuϕ
    kuϕ = policy.kuϕ
    Ku = policy.Ku
    ku = policy.ku
    Kϕ = policy.Kϕ
    kϕ = policy.kϕ
    Kvl = policy.Kvl
    kvl = policy.kvl
    Kvu = policy.Kvu
    kvu = policy.kvu
    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    
    x, u, h, il, iu = primal_trajectories(problem, mode=mode)
    ϕ, vl, vu = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = 0.

    S = nothing
    
    while reg <= options.reg_max
        data.status = true
        Vxx[N] .= lxx[N]
        Vx[N] .= lx[N]
        
        for t = N-1:-1:1
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
    
            # Qux = lux + fu' * Vxx * fx
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Qux[t], policy.ux_tmp[t], fx[t])
            Qux[t] .+= lux[t]
            
            # apply second order terms to Q for full DDP, i.e., Vx * fxx, Vx * fuu, Vx * fxu
            if !options.quasi_newton
                hessian_vector_prod!(fxx[t], fux[t], fuu[t], problem.model.dynamics[t], x[t], u[t], Vx[t+1])
                # display(fxx[t])
                Qxx[t] .+= fxx[t]
                Qux[t] .+= fux[t]
                Quu[t] .+= fuu[t]
                
                hessian_vector_prod!(hxx[t], hux[t], huu[t], constr_data.constraints[t], x[t], u[t], ϕ[t])
            end
            # display(Vx[t+1])
            
            # setup linear system in backward pass
            policy.lhs_tl[t] .= Quu[t]
            policy.lhs_tr[t] .= transpose(hu[t])
            policy.lhs_bl[t] .= hu[t]
            fill!(policy.lhs_br[t], 0.0)
            add_primal_dual!(policy.lhs_tl[t], il[t], iu[t], vl[t], vu[t])
            policy.lhs_tl[t] .+= huu[t]

            policy.rhs_t[t] .= -Qu[t]
            policy.rhs_b[t] .= -h[t]

            policy.rhs_x_t[t] .= -Qux[t]
            policy.rhs_x_b[t] .= -hx[t]
            policy.rhs_x_t[t] .-= hux[t]

            # inertia calculation and correction

            policy.lhs_tl[t][diagind(policy.lhs_tl[t])] .+= reg
            policy.lhs_br[t][diagind(policy.lhs_br[t])] .-= δ_c
            try
                S = bunchkaufman!(policy.lhs[t])
            catch
                if iszero(δ_c)
                    δ_c = options.δ_c * μ^options.κ_c
                end
                if iszero(reg) # initial setting of regularisation
                    reg = (data.reg_last == 0.0) ? options.reg_1 : max(options.reg_min, options.κ_w_m * data.reg_last)
                else
                    reg = (data.reg_last == 0.0) ? options.κ_̄w_p * reg : options.κ_w_p * reg
                end
                data.status = false
                break
            end
            np, nn, nz = inertia(S.D)
            num_constr = length(h[t])
            num_actions = length(u[t])
            if np != num_actions || nn != num_constr
                if nz > 0 && iszero(δ_c)
                    δ_c = options.δ_c * μ^options.κ_c
                end
                if iszero(reg) # initial setting of regularisation
                    reg = (data.reg_last == 0.0) ? options.reg_1 : max(options.reg_min, options.κ_w_m * data.reg_last)
                else
                    reg = (data.reg_last == 0.0) ? options.κ_̄w_p * reg : options.κ_w_p * reg
                end
                data.status = false
                break
            end
            
            # update gains for controls and eq. duals
            kuϕ[t] .= S \ policy.rhs[t]
            Kuϕ[t] .= S \ policy.rhs_x[t]

            # update gains for ineq. duals
            gains_ineq!(kvl[t], Kvl[t], il[t], vl[t], ku[t], Ku[t], μ)
            gains_ineq!(kvu[t], Kvu[t], iu[t], vu[t], ku[t], Ku[t], μ)
            
            # Update return function approx. for next timestep 
            # Vxx = Q̂xx + Q̂ux' * Ku + Ku * Q̂ux' + Ku' Q̂uu' * Ku
            # add_barrier_hess!(Quu[t], il[t], iu[t], μ)
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

function add_barrier_hess!(Quu, ineq_lower, ineq_upper, μ)
    m = length(ineq_lower)
    for i = 1:m
        if !isinf(ineq_lower[i])
            Quu[i, i] -= μ / ineq_lower[i] ^ 2
        end
        if !isinf(ineq_upper[i])
            Quu[i, i] -= μ / ineq_upper[i] ^ 2
        end
    end
end

function inertia(D; tol=1e-12)
    n::Int = size(D)[1]
    i::Int = 1
    pos::Int = 0
    neg::Int = 0
    zr::Int = 0
    while i <= n
        if i < n && abs(D[i+1, i]) > tol
            pos += 1
            neg += 1
            i += 2
        elseif abs(D[i, i]) > tol
            if D[i, i] > 0
                pos += 1
            else
                neg += 1
            end
            i += 1
        else
            zr += 1
            i += 1
        end
    end
    return pos, neg, zr
end

function gains_ineq!(k, K, ineq, duals, ku, Ku, μ)
    m = length(ineq)
    for i = 1:m
        if !isinf(ineq[i])
            k[i] = μ / ineq[i] - duals[i] - duals[i] / ineq[i] * ku[i]
        else
            k[i] = 0.
        end
        if !isinf(ineq[i])
            K[i, :] = duals[i] / ineq[i] * Ku[i, :]
        else
            fill!(K[i, :], 0.0)
        end
    end
end
