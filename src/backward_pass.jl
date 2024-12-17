function backward_pass!(update_rule::UpdateRuleData{T}, problem::ProblemData{T}, data::SolverData{T},
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
    # Hamiltonian approximation
    Q̂x = update_rule.hamiltonian.gradient_state
    Q̂u = update_rule.hamiltonian.gradient_control
    Q̂xx = update_rule.hamiltonian.hessian_state_state
    Q̂uu = update_rule.hamiltonian.hessian_control_control
    Q̂ux = update_rule.hamiltonian.hessian_control_state
    # Update rule parameters
    α = update_rule.parameters.α
    β = update_rule.parameters.β
    ψ = update_rule.parameters.ψ
    ω = update_rule.parameters.ω
    χl = update_rule.parameters.χl
    ζl = update_rule.parameters.ζl
    χu = update_rule.parameters.χu
    ζu = update_rule.parameters.ζu
    # Value function
    V̂x = update_rule.value.gradient
    V̂xx = update_rule.value.hessian
    # Bound constraints
    bounds = problem.bounds

    bl1 = update_rule.bl_tmp1
    bl2 = update_rule.bl_tmp2
    bu1 = update_rule.bu_tmp1
    bu2 = update_rule.bu_tmp2
    
    x, u, h = primal_trajectories(problem, mode=mode)
    ϕ, zl, zu = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = 0.
    reg = 0.0
    
    while reg <= options.reg_max
        data.status = 0
        V̂xx[N] .= lxx[N]
        V̂x[N] .= lx[N]
        
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

            # Q̂x = Lx + fx' * V̂x
            Q̂x[t] .= lx[t]
            mul!(Q̂x[t], transpose(fx[t]), V̂x[t+1], 1.0, 1.0)
            mul!(Q̂x[t], transpose(hx[t]), ϕ[t], 1.0, 1.0)

            # Q̂u = Lu + fu' * V̂x
            Q̂u[t] .= lu[t]
            mul!(Q̂u[t], transpose(fu[t]), V̂x[t+1], 1.0, 1.0)
            @views Q̂u[t][bt.indices_lower] .-= bl1[t]
            @views Q̂u[t][bt.indices_upper] .+= bu1[t]
            mul!(Q̂u[t], transpose(hu[t]), ϕ[t], 1.0, 1.0)
            
            # Q̂xx = Lxx + fx' * V̂xx * fx + V̂xx ⋅ fxx
            mul!(update_rule.xx_tmp[t], transpose(fx[t]), V̂xx[t+1])
            mul!(Q̂xx[t], update_rule.xx_tmp[t], fx[t])
            Q̂xx[t] .+= lxx[t]
    
            # Q̂uu = ℓuu + ϕ ⋅ huu + Σ + fu' * V̂xx * fu + V̂x ⋅ fuu
            mul!(update_rule.ux_tmp[t], transpose(fu[t]), V̂xx[t+1])
            mul!(Q̂uu[t], update_rule.ux_tmp[t], fu[t])
            Q̂uu[t] .+= luu[t]
            for (i1, i) in enumerate(bt.indices_lower)
                Q̂uu[t][i, i] += bl2[t][i1]
            end
            for (i1, i) in enumerate(bt.indices_upper)
                Q̂uu[t][i, i] += bu2[t][i1]
            end
    
            # Q̂xu = Lxu + fu' * V̂xx * fx + V̂x ⋅ fxu
            mul!(update_rule.ux_tmp[t], transpose(fu[t]), V̂xx[t+1])
            mul!(Q̂ux[t], update_rule.ux_tmp[t], fx[t])
            Q̂ux[t] .+= lux[t]
            
            # apply second order tensor contraction terms to Q̂uu, Q̂ux, Q̂xx
            if !options.quasi_newton
                fn_eval_time_ = time()
                tensor_contraction!(vfxx[t], vfux[t], vfuu[t], problem.model.dynamics[t], x[t], u[t], V̂x[t+1])
                data.fn_eval_time += time() - fn_eval_time_
                Q̂xx[t] .+= vfxx[t]
                Q̂ux[t] .+= vfux[t]
                Q̂uu[t] .+= vfuu[t]

                Q̂uu[t] .+= vhuu[t]
                Q̂ux[t] .+= vhux[t]
                Q̂xx[t] .+= vhxx[t]
            end
            
            # setup linear system in backward pass
            update_rule.lhs_tl[t] .= Q̂uu[t]
            update_rule.lhs_tr[t] .= transpose(hu[t])
            fill!(update_rule.lhs_br[t], 0.0)

            α[t] .= Q̂u[t]
            α[t] .*= -1.0
            ψ[t] .= h[t]
            ψ[t] .*= -1.0
            β[t] .= Q̂ux[t]
            β[t] .*= -1.0
            ω[t] .= hx[t]
            ω[t] .*= -1.0

            # inertia calculation and correction (regularisation)
            if reg > 0.0
                for i in 1:num_control
                    update_rule.lhs_tl[t][i, i] += reg
                end
            end
            if δ_c > 0.0
                for i in 1:num_constr
                    update_rule.lhs_br[t][i, i] -= δ_c
                end
            end

            bk, data.status, reg, δ_c = inertia_correction!(update_rule.kkt_matrix_ws[t], update_rule.lhs[t], update_rule.D_cache[t],
                        num_control, μ, reg, data.reg_last, options)

            data.status != 0 && break

            ldiv!(bk, update_rule.parameters.eq[t])

            # update parameters of update rule for ineq. dual variables 
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
            mul!(update_rule.ux_tmp[t], Q̂uu[t], β[t])
            mul!(V̂xx[t], transpose(β[t]), Q̂ux[t])

            mul!(V̂xx[t], transpose(Q̂ux[t]), β[t], 1.0, 1.0)
            V̂xx[t] .+= Q̂xx[t]
            mul!(V̂xx[t], transpose(β[t]), update_rule.ux_tmp[t], 1.0, 1.0)
            V̂xx[t] .= Symmetric(V̂xx[t])

            # Vx = Q̂x + β' * Q̂u + [Q̂uu β + Q̂ux]^T α
            update_rule.ux_tmp[t] .+= Q̂ux[t]
            mul!(V̂x[t], transpose(update_rule.ux_tmp[t]), α[t])
            mul!(V̂x[t], transpose(β[t]), Q̂u[t], 1.0, 1.0)
            V̂x[t] .+= Q̂x[t]
        end
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find positive definite iteration matrix."))
end


