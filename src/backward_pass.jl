function backward_pass!(policy::PolicyData{T}, problem::ProblemData{T}, data::SolverData{T},
            options::Options{T}; mode=:nominal, verbose::Bool=false) where T
    N = problem.horizon
    reg::T = 0.0

    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_control
    # Tensor contraction of system dynamics (DDP)
    vfxx = problem.model.vfxx
    vfux = problem.model.vfux
    vfuu = problem.model.vfuu
    # Jacobian caches
    hx = problem.constraints_data.jacobian_state
    hu = problem.constraints_data.jacobian_control
    # Tensor contraction of constraints (DDP)
    vhxx = problem.constraints_data.vhxx
    vhux = problem.constraints_data.vhux
    vhuu = problem.constraints_data.vhuu
    # Cost gradients
    lx = problem.cost_data.gradient_state
    lu = problem.cost_data.gradient_control
    # Cost hessians
    lxx = problem.cost_data.hessian_state_state
    luu = problem.cost_data.hessian_control_control
    lux = problem.cost_data.hessian_control_state
    # control-Value function approximation
    Qx = policy.hamiltonian.gradient_state
    Qu = policy.hamiltonian.gradient_control
    Qxx = policy.hamiltonian.hessian_state_state
    Quu = policy.hamiltonian.hessian_control_control
    Qux = policy.hamiltonian.hessian_control_state
    # Feedback gains (linear feedback policy)
    α = policy.gains_data.α
    β = policy.gains_data.β
    ψ = policy.gains_data.ψ
    ω = policy.gains_data.ω
    χl = policy.gains_data.χl
    ζl = policy.gains_data.ζl
    χu = policy.gains_data.χu
    ζu = policy.gains_data.ζu
    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    # Bound constraints
    bounds = problem.bounds

    bl1 = policy.bl_tmp1
    bl2 = policy.bl_tmp2
    bu1 = policy.bu_tmp1
    bu2 = policy.bu_tmp2
    
    x, u, h = primal_trajectories(problem, mode=mode)
    ϕ, zl, zu = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = 0.
    reg = 0.0
    
    while reg <= options.reg_max
        data.status = 0
        Vxx[N] .= lxx[N]
        Vx[N] .= lx[N]
        
        for t = N-1:-1:1
            num_control = length(u[t])
            num_constr = length(h[t])

            bt = bounds[t]

            bl1[t] .= @views u[t][bt.indices_lower]
            bl1[t] .-= @views bt.lower[bt.indices_lower]
            bl1[t] .= inv.(bl1[t])
            bl2[t] .= bl1[t]  # σ_L
            bl2[t] .*= @views zl[t][bt.indices_lower]
            bl1[t] .*= μ  # μ / (u - ul)

            bu1[t] .= @views bt.upper[bt.indices_upper]
            bu1[t] .-= @views u[t][bt.indices_upper]
            bu1[t] .= inv.(bu1[t])
            bu2[t] .= bu1[t]  # σ_Uq
            bu2[t] .*= @views zu[t][bt.indices_upper]
            bu1[t] .*= μ  # μ / (uu - u)

            # Qx = lx + fx' * Vx
            Qx[t] .= lx[t]
            mul!(Qx[t], transpose(fx[t]), Vx[t+1], 1.0, 1.0)
            mul!(Qx[t], transpose(hx[t]), ϕ[t], 1.0, 1.0)

            # Qu = lu + fu' * Vx + μ log(u - ul) + μ log(uu - u)
            Qu[t] .= lu[t]
            mul!(Qu[t], transpose(fu[t]), Vx[t+1], 1.0, 1.0)
            @views Qu[t][bt.indices_lower] .-= bl1[t]
            @views Qu[t][bt.indices_upper] .+= bu1[t]
            mul!(Qu[t], transpose(hu[t]), ϕ[t], 1.0, 1.0)
            
            # Qxx = lxx + fx' * Vxx * fx
            mul!(policy.xx_tmp[t], transpose(fx[t]), Vxx[t+1])
            mul!(Qxx[t], policy.xx_tmp[t], fx[t])
            Qxx[t] .+= lxx[t]
    
            # Quu = luu + fu' * Vxx * fu + Σ
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Quu[t], policy.ux_tmp[t], fu[t])
            Quu[t] .+= luu[t]
            for (i1, i) in enumerate(bt.indices_lower)
                Quu[t][i, i] += bl2[t][i1]
            end
            for (i1, i) in enumerate(bt.indices_upper)
                Quu[t][i, i] += bu2[t][i1]
            end
    
            # Qux = lux + fu' * Vxx * fx
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Qux[t], policy.ux_tmp[t], fx[t])
            Qux[t] .+= lux[t]
            
            # apply second order terms to Q for full DDP, i.e., Vx * fxx, Vx * fuu, Vx * fxu
            if !options.quasi_newton
                tensor_contraction!(vfxx[t], vfux[t], vfuu[t], problem.model.dynamics[t], x[t], u[t], Vx[t+1])
                Qxx[t] .+= vfxx[t]
                Qux[t] .+= vfux[t]
                Quu[t] .+= vfuu[t]
            end
            if !options.quasi_newton
                Quu[t] .+= vhuu[t]
                Qux[t] .+= vhux[t]
                Qxx[t] .+= vhxx[t]
            end
            # setup linear system in backward pass
            policy.lhs_tl[t] .= Quu[t]
            policy.lhs_tr[t] .= transpose(hu[t])
            fill!(policy.lhs_br[t], 0.0)
            # if !options.quasi_newton
            #     policy.lhs_tl[t] .+= vhuu[t]
            # end
            policy.lhs[t] .= Symmetric(policy.lhs[t])

            α[t] .= Qu[t]
            α[t] .*= -1.0
            # mul!(α[t], transpose(hu[t]), ϕ[t], -1.0, 1.0)
            ψ[t] .= h[t]
            ψ[t] .*= -1.0
            β[t] .= Qux[t]
            β[t] .*= -1.0
            ω[t] .= hx[t]
            ω[t] .*= -1.0
            # if !options.quasi_newton
            #     β[t] .-= vhux[t]
            # end

            # inertia calculation and correction
            if reg > 0.0
                for i in 1:num_control
                    policy.lhs_tl[t][i, i] += reg
                end
            end
            if δ_c > 0.0
                for i in 1:num_constr
                    policy.lhs_br[t][i, i] -= δ_c
                end
            end

            bk, data.status, reg, δ_c = inertia_correction!(policy.kkt_matrix_ws[t], policy.lhs[t], policy.D_cache[t],
                        num_control, μ, reg, data.reg_last, options)

            data.status != 0 && break

            ldiv!(bk, policy.gains_data.gains[t])

            # update gains for ineq. dual variables 
            # χ_L =  μ inv.(u) - z - σ_L .* α 
            # ζ_L =  - σ_L .* β 
            # χ_U =  μ inv.(u) - z - σ_U .* α 
            # ζ_U =  σ_U .* β
                        
            χlt, χut = χl[t], χu[t]
            ζlt, ζut = ζl[t], ζu[t]
            αt, βt = α[t], β[t]
            bl1t, bl2t = bl1[t], bl2[t]
            bu1t, bu2t = bu1[t], bu2[t]
            zlt, zut = zl[t], zu[t]
            for (i1, i) in enumerate(bt.indices_lower)
                # feedforward
                χlti = @views χlt[i, :]
                χlti .= @views αt[i, :]
                χlti .*= @views bl2t[i1]
                χlti .*= -1.0
                χlti .-= @views zlt[i]
                χlti .+= @views bl1t[i1]
                # feedback
                ζlti = @views ζlt[i, :]
                ζlti .= @views βt[i, :]
                ζlti .*= @views bl2t[i1]
                ζlti .*= -1.0
            end
            
            for (i1, i) in enumerate(bt.indices_upper)
                # feedforward
                χuti = @views χut[i, :]
                χuti .= @views αt[i, :]
                χuti .*= @views bu2t[i1]
                χuti .-= @views zut[i]
                χuti .+= @views bu1t[i1]
                # feedback
                ζuti = @views ζut[i, :]
                ζuti .= @views βt[i, :]
                ζuti .*= @views bu2t[i1]
            end

            # Update return function approx. for next timestep 
            # Vxx = Q̂xx + Q̂ux' * β + β * Q̂ux' + β' Q̂uu' * β
            mul!(policy.ux_tmp[t], Quu[t], β[t])
            mul!(Vxx[t], transpose(β[t]), Qux[t])

            mul!(Vxx[t], transpose(Qux[t]), β[t], 1.0, 1.0)
            Vxx[t] .+= Qxx[t]
            mul!(Vxx[t], transpose(β[t]), policy.ux_tmp[t], 1.0, 1.0)
            Vxx[t] .= Symmetric(Vxx[t])

            # Vx = Q̂x + β' * Q̂u + [Q̂uu β + Q̂ux]^T α
            policy.ux_tmp[t] .+= Qux[t]
            mul!(Vx[t], transpose(policy.ux_tmp[t]), α[t])
            mul!(Vx[t], transpose(β[t]), Qu[t], 1.0, 1.0)
            Vx[t] .+= Qx[t]
        end
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find positive definite iteration matrix."))
end


