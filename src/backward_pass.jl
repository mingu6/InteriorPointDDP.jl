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
    Q̃u = update_rule.Q̃u
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

    u_tmp1 = update_rule.u_tmp1
    u_tmp2 = update_rule.u_tmp2
    uxnext_tmp = update_rule.uxnext_tmp
    xx_tmp = update_rule.xx_tmp
    
    x, u, h, il, iu = primal_trajectories(problem, mode=mode)
    ϕ, zl, zu = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = T(0.0)
    reg = T(0.0)
    
    while reg <= options.reg_max
        data.status = 0
        V̂xx[N] .= lxx[N]
        V̂x[N] .= lx[N]
        
        for t = N-1:-1:1
            num_control = length(u[t])
            num_constr = length(h[t])

            u_tmp1[t] .= inv.(il[t])
            u_tmp2[t] .= inv.(iu[t])

            # Q̂x = Lx + fx' * V̂x
            Q̂x[t] .= lx[t]
            mul!(Q̂x[t], transpose(fx[t]), V̂x[t+1], 1.0, 1.0)
            mul!(Q̂x[t], transpose(hx[t]), ϕ[t], 1.0, 1.0)

            # Q̂u = Lu + fu' * V̂x
            Q̃u[t] .= lu[t]
            mul!(Q̃u[t], transpose(fu[t]), V̂x[t+1], 1.0, 1.0)
            mul!(Q̃u[t], transpose(hu[t]), ϕ[t], 1.0, 1.0)

            χl[t] .= u_tmp1[t]
            χl[t] .*= μ
            χu[t] .= u_tmp2[t]
            χu[t] .*= μ

            Q̂u[t] .= χl[t]      # barrier term gradient
            Q̂u[t] .*= -1.0      # barrier term gradient
            Q̂u[t] .+= χu[t]     # barrier term gradient
            Q̂u[t] .+= Q̃u[t]

            Q̃u[t] .-= zl[t]
            Q̃u[t] .+= zu[t]
            
            # Q̂xx = Lxx + fx' * V̂xx * fx + V̂xx ⋅ fxx
            mul!(xx_tmp[t], transpose(fx[t]), V̂xx[t+1])
            mul!(Q̂xx[t], xx_tmp[t], fx[t])
            Q̂xx[t] .+= lxx[t]
    
            # Q̂uu = luu + ϕ ⋅ huu + Σ + fu' * V̂xx * fu + V̂x ⋅ fuu
            u_tmp1[t] .*= zl[t]   # Σ^L
            u_tmp2[t] .*= zu[t]   # Σ^U
            fill!(Q̂uu[t], 0.0)
            for i = 1:num_control
                Q̂uu[t][i, i] = u_tmp1[t][i] + u_tmp2[t][i]
            end
            
            mul!(uxnext_tmp[t], transpose(fu[t]), V̂xx[t+1])
            mul!(Q̂uu[t], uxnext_tmp[t], fu[t], 1.0, 1.0)
            Q̂uu[t] .+= luu[t]
    
            # Q̂xu = Lxu + fu' * V̂xx * fx + V̂x ⋅ fxu
            mul!(update_rule.uxnext_tmp[t], transpose(fu[t]), V̂xx[t+1])
            mul!(Q̂ux[t], update_rule.uxnext_tmp[t], fx[t])
            Q̂ux[t] .+= lux[t]
            
            # apply second order tensor contraction terms to Q̂uu, Q̂ux, Q̂xx
            if !options.quasi_newton
                fn_eval_time_ = time()
                tensor_contraction!(vfxx[t], vfux[t], vfuu[t], problem.model.dynamics[t], x[t], u[t], V̂x[t+1])
                data.fn_eval_time += time() - fn_eval_time_
            else
                data.k != 0 && sr1_update!(problem, update_rule, data, t; mode=mode)
            end

            Q̂xx[t] .+= vfxx[t]
            Q̂ux[t] .+= vfux[t]
            Q̂uu[t] .+= vfuu[t]

            Q̂uu[t] .+= vhuu[t]
            Q̂ux[t] .+= vhux[t]
            Q̂xx[t] .+= vhxx[t]
            
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

            # update parameters of update rule for ineq. dual variables, i.e., 

            # χ_L =  μ inv.(u - u^L) - z^L - Σ^L * α 
            # ζ_L =  - Σ^L * β 
            # χ_U =  μ inv.(u^U - u) - z^U + Σ^U .* α 
            # ζ_U =  Σ^U .* β

            # see update above for Q̂u[t] for first part of χ^L[t] χ^U[t]

            ζl[t] .= β[t]
            ζl[t] .*= u_tmp1[t]
            ζl[t] .*= -1.0

            u_tmp1[t] .*= α[t]
            χl[t] .-= zl[t]
            χl[t] .-= u_tmp1[t]

            ζu[t] .= β[t]
            ζu[t] .*= u_tmp2[t]

            u_tmp2[t] .*= α[t]
            χu[t] .-= zu[t]
            χu[t] .+= u_tmp2[t]

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

function sr1_update!(problem::ProblemData{T}, update_rule::UpdateRuleData{T}, data::SolverData{T},
            t::Int64; mode=:nominal) where T
    # TODO: update constraint and dynamics together?
    # TODO: init SR1, set parameters for checks
    # TODO: 2-norm for check?
    
    x, u, _, _, _ = primal_trajectories(problem, mode=mode)
    ϕ, _, _ = dual_trajectories(problem, mode=mode)
    x_prev = problem.previous_states
    u_prev = problem.previous_controls

    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_control
    fx_prev = problem.previous_fx
    fu_prev = problem.previous_fu
    
    vfxx = problem.model.vfxx
    vfux = problem.model.vfux
    vfuu = problem.model.vfuu
    
    hx = problem.constraints_data.jacobian_state
    hu = problem.constraints_data.jacobian_control
    hx_prev = problem.previous_hx
    hu_prev = problem.previous_hu
    
    vhxx = problem.constraints_data.vhxx
    vhux = problem.constraints_data.vhux
    vhuu = problem.constraints_data.vhuu

    V̂x = update_rule.value.gradient

    u_tmp = update_rule.u_tmp
    u_tmp3 = update_rule.u_tmp3
    x_tmp = update_rule.x_tmp
    x_tmp1 = update_rule.x_tmp1
    xx_tmp = update_rule.xx_tmp
    uu_tmp = update_rule.uu_tmp
    ux_tmp = update_rule.ux_tmp
    hx_tmp = update_rule.hx_tmp
    hu_tmp = update_rule.hu_tmp
    uxnext_tmp = update_rule.uxnext_tmp

    r = 1e-8

    # δx-, δu-
    u_tmp[t] .= u[t]
    u_tmp[t] .-= u_prev[t]
    x_tmp[t] .= x[t]
    x_tmp[t] .-= x_prev[t]

    # ## dynamics Hessian terms

    # SR1 residual for rank 1 update, i.e., y - E^- (δx-, δu-)
    mul!(x_tmp1[t], vfxx[t], x_tmp[t])
    mul!(x_tmp1[t], transpose(vfux[t]), u_tmp[t], 1.0, 1.0)
    x_tmp1[t] .*= -1.0

    mul!(u_tmp3[t], vfuu[t], u_tmp[t])
    mul!(u_tmp3[t], vfux[t], x_tmp[t], 1.0, 1.0)
    u_tmp3[t] .*= -1.0

    uxnext_tmp[t] .= transpose(fu[t])
    uxnext_tmp[t] .-= transpose(fu_prev[t])
    mul!(u_tmp3[t], uxnext_tmp[t], V̂x[t+1], 1.0, 1.0)

    xx_tmp[t] .= transpose(fx[t])
    xx_tmp[t] .-= transpose(fx_prev[t])
    mul!(x_tmp1[t], xx_tmp[t], V̂x[t+1], 1.0, 1.0)

    # apply rank 1 SR1 update
    sr1_denom = dot(x_tmp1[t], x_tmp[t])
    sr1_denom += dot(u_tmp3[t], u_tmp[t])

    # check if update is large enough
    if abs(sr1_denom) > r * (norm(x_tmp1[t], 1) + norm(u_tmp3[t], 1)) * (norm(x_tmp[t], 1) + norm(u_tmp[t], 1))
        xx_tmp[t] .= x_tmp1[t]
        xx_tmp[t] .*= transpose(x_tmp1[t])  # TODO: replace with mul! ?
        xx_tmp[t] ./= sr1_denom
        vfxx[t] .+= xx_tmp[t]

        uu_tmp[t] .= u_tmp3[t]
        uu_tmp[t] .*= transpose(u_tmp3[t])
        uu_tmp[t] ./= sr1_denom
        vfuu[t] .+= uu_tmp[t]

        ux_tmp[t] .= u_tmp3[t]
        ux_tmp[t] .*= transpose(x_tmp1[t])
        ux_tmp[t] ./= sr1_denom
        vfux[t] .+= ux_tmp[t]
    end

    # ## constraint Hessian terms

    # SR1 residual for rank 1 update, i.e., y - E^- (δx-, δu-)
    mul!(x_tmp1[t], vhxx[t], x_tmp[t])
    mul!(x_tmp1[t], transpose(vhux[t]), u_tmp[t], 1.0, 1.0)
    x_tmp1[t] .*= -1.0

    mul!(u_tmp3[t], vhuu[t], u_tmp[t])
    mul!(u_tmp3[t], vhux[t], x_tmp[t], 1.0, 1.0)
    u_tmp3[t] .*= -1.0

    hu_tmp[t] .= hu[t]
    hu_tmp[t] .-= hu_prev[t]
    mul!(u_tmp3[t], transpose(hu_tmp[t]), ϕ[t], 1.0, 1.0)

    hx_tmp[t] .= hx[t]
    hx_tmp[t] .-= hx_prev[t]
    mul!(x_tmp1[t], transpose(hx_tmp[t]), ϕ[t], 1.0, 1.0)

    # apply rank 1 SR1 update
    sr1_denom = dot(x_tmp1[t], x_tmp[t])
    sr1_denom += dot(u_tmp3[t], u_tmp[t])

    if abs(sr1_denom) > r * (norm(x_tmp1[t], 1) + norm(u_tmp3[t], 1)) * (norm(x_tmp[t], 1) + norm(u_tmp[t], 1))
        xx_tmp[t] .= x_tmp1[t]
        xx_tmp[t] .*= transpose(x_tmp1[t])
        xx_tmp[t] ./= sr1_denom
        vhxx[t] .+= xx_tmp[t]

        uu_tmp[t] .= u_tmp3[t]
        uu_tmp[t] .*= transpose(u_tmp3[t])
        uu_tmp[t] ./= sr1_denom
        vhuu[t] .+= uu_tmp[t]

        ux_tmp[t] .= u_tmp3[t]
        ux_tmp[t] .*= transpose(x_tmp1[t])
        ux_tmp[t] ./= sr1_denom
        vhux[t] .+= ux_tmp[t]
    end
    return nothing
end
