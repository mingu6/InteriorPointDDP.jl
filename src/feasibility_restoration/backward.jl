function fr_backward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options, 
                           u_R::Vector{T}, D_R::Vector{T}, μ::Float64, ρ::Float64, ζ::Float64;
                           mode=:nominal, verbose::Bool=false) where T
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
    Kϕ = policy.gains_main.Kϕ
    kϕ = policy.gains_main.kϕ
    kvl = policy.gains_main.kvl
    kvu = policy.gains_main.kvu
    Kvl = policy.gains_main.Kvl
    Kvu = policy.gains_main.Kvu

    kp = policy.kp
    kn = policy.kn
    Kp = policy.Kp
    Kn = policy.Kn
    kvp = policy.kvp
    kvn = policy.kvn
    Kvp = policy.Kvp
    Kvn = policy.Kvn
    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    
    x, u, h, il, iu = primal_trajectories(problem, mode=mode)
    ϕ, vl, vu = dual_trajectories(problem, mode=mode)
    p, n, vp, vn = fr_trajectories(problem, mode=mode)

    # display(problem.nominal_p[end] ./ problem.nominal_vp[end])
    # display(problem.nominal_n[end] ./ problem.nominal_vn[end])
    # display(- p[end] ./ vp[end] - n[end] ./ vn[end])
    while reg <= options.reg_max
        data.status = true
        fill!(Vxx[N], 0.0)
        fill!(Vx[N], 0.0)
        
        for t = N-1:-1:1
            num_actions = length(ku[t])
            # Qx = lx + fx' * Vx
            mul!(Qx[t], transpose(fx[t]), Vx[t+1])

            # Qu = lu + fu' * Vx
            Qu[t] .= ζ .* D_R[t].^2 .* (u[t] - u_R[t])
            mul!(Qu[t], transpose(fu[t]), Vx[t+1], 1.0, 1.0)
            add_barrier_grad!(Qu[t], il[t], iu[t], μ)

            # Qxx = lxx + fx' * Vxx * fx
            mul!(policy.xx_tmp[t], transpose(fx[t]), Vxx[t+1])
            mul!(Qxx[t], policy.xx_tmp[t], fx[t])
    
            # Quu = luu + fu' * Vxx * fu
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Quu[t], policy.ux_tmp[t], fu[t])
            add_primal_dual!(Quu[t], il[t], iu[t], vl[t], vu[t])
            Quu[t][diagind(Quu[t])] .+= ζ * D_R[t] .^ 2
    
            # Qux = lux + fu' * Vxx * fx
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Qux[t], policy.ux_tmp[t], fx[t])

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
            policy.lhs_br[t][diagind(policy.lhs_br[t])] .= - p[t] ./ vp[t] - n[t] ./ vn[t]

            policy.rhs_t[t] .= -Qu[t]
            mul!(policy.rhs_t[t], transpose(hu[t]), ϕ[t], -1.0, 1.0)
            policy.rhs_b[t] .= -h[t] - ρ .* ((μ .- p[t]) ./ vp[t] + (μ .- n[t]) ./ vn[t])

            fill!(policy.rhs_x[t], 0.0)
            policy.rhs_x_b[t] .= -hx[t]
            if !options.quasi_newton
                policy.rhs_x_t[t] .-= hux[t]
            end

            # if t == 1
            #     display(policy.lhs[t])
            #     display(policy.rhs[t])
            #     display(policy.rhs_x[t])
            # end

            # inertia correction
            policy.lhs_tl[t][diagind(policy.lhs_tl[t])] .+= reg

            policy.lhs_bk[t], status, reg, _ = inertia_correction!(policy.lhs[t], num_actions,
                        μ, reg, data.reg_last, options; rook=true)
            !status && break

            kuϕ[t] .= policy.lhs_bk[t] \ policy.rhs[t]
            Kuϕ[t] .= policy.lhs_bk[t] \ policy.rhs_x[t]

            # if t == 1
            #     display(kuϕ[t])
            #     display(Kuϕ[t])
            #     display(u[t])
            #     display(il[t])
            # end

            # update gains for ineq. duals
            gains_ineq!(kvl[t], kvu[t], Kvl[t], Kvu[t], il[t], iu[t], vl[t], vu[t], ku[t], Ku[t], μ)

            # update gains for slack variables and duals
            kp[t] .= (μ .+ p[t] .* (ϕ[t] + kϕ[t]) - ρ .* p[t]) ./ vp[t]
            kn[t] .= (μ .- n[t] .* (ϕ[t] + kϕ[t]) - ρ .* n[t]) ./ vn[t]
            Kp[t] .= p[t] ./ vp[t] .* Kϕ[t]
            Kn[t] .= -n[t] ./ vn[t] .* Kϕ[t]

            kvp[t] .= μ ./ p[t] - vp[t] - vp[t] ./ p[t] .* kp[t]
            kvn[t] .= μ ./ n[t] - vn[t] - vn[t] ./ n[t] .* kn[t]
            Kvp[t] .= -vp[t] ./ p[t] .* Kp[t]
            Kvn[t] .= -vn[t] ./ n[t] .* Kn[t]

            # if t == 1
            #     display(kp[t])
            #     display(kn[t])
            #     display(kvp[t])
            #     display(kvn[t])
            #     println(" ")
            #     display(p[t])
            #     display(n[t])
            #     display(vp[t])
            #     display(vn[t])
            # end
            
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
    !data.status && (verbose && (@warn "(FR) Backward pass failure, unable to find positive definite iteration matrix."))
end

