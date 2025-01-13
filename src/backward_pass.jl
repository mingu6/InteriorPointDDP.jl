using IterativeRefinement

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
    # Objective gradients
    lx = problem.objective_data.gradient_state
    lu = problem.objective_data.gradient_control
    # Objective hessians
    lxx = problem.objective_data.hessian_state_state
    luu = problem.objective_data.hessian_control_control
    lux = problem.objective_data.hessian_control_state
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
    
    x, u, h, il, iu = primal_trajectories(problem, mode=mode)
    ϕ, zl, zu = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = 0.
    reg = 0.0

    bounds = problem.bounds
    
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
            mul!(update_rule.xx_tmp[t], transpose(fx[t]), V̂xx[t+1])
            mul!(Q̂xx[t], update_rule.xx_tmp[t], fx[t])
            Q̂xx[t] .+= lxx[t]
    
            # Q̂uu = luu + ϕ ⋅ huu + Σ + fu' * V̂xx * fu + V̂x ⋅ fuu
            u_tmp1[t] .*= zl[t]   # Σ^L
            u_tmp2[t] .*= zu[t]   # Σ^U
            fill!(Q̂uu[t], 0.0)
            for i = 1:num_control
                Q̂uu[t][i, i] = u_tmp1[t][i] + u_tmp2[t][i]
            end

            mul!(update_rule.ux_tmp[t], transpose(fu[t]), V̂xx[t+1])
            mul!(Q̂uu[t], update_rule.ux_tmp[t], fu[t], 1.0, 1.0)
            Q̂uu[t] .+= luu[t]
    
            # Q̂xu = Lxu + fu' * V̂xx * fx + V̂x ⋅ fxu
            mul!(update_rule.ux_tmp[t], transpose(fu[t]), V̂xx[t+1])
            mul!(Q̂ux[t], update_rule.ux_tmp[t], fx[t])
            Q̂ux[t] .+= lux[t]

            # Q̃uu
            Q̃uu = zeros(num_control, num_control)
            mul!(Q̃uu, update_rule.ux_tmp[t], fu[t], 1.0, 1.0)
            Q̃uu .+= luu[t]
            
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

                Q̃uu .+= vfuu[t]
                Q̃uu .+= vhuu[t]
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

            reg1 = copy(reg)
            δ_c1 = copy(δ_c)

            bk, data.status, reg, δ_c = inertia_correction!(update_rule.kkt_matrix_ws[t], update_rule.lhs[t], update_rule.D_cache[t],
                        num_control, μ, reg, data.reg_last, options)

            data.status != 0 && break

            ldiv!(bk, update_rule.parameters.eq[t])


            # iterative refinement - setup uncondensed asymmetric system

            # num_ineq = bounds[t].num_lower + bounds[t].num_upper
            # ind_l = bounds[t].indices_lower
            # ind_u = bounds[t].indices_upper
            # nl = bounds[t].num_lower
            # nu = bounds[t].num_upper
            # il_m = zeros(nl, nl)
            # iu_m = zeros(nu, nu)
            # zl_m = zeros(nl, num_control)
            # zu_m = zeros(nu, num_control)
            # eye_l = zeros(num_control, nl)
            # eye_u = zeros(num_control, nu)

            # for l in 1:nl
            #     zl_m[l, ind_l[l]] = zl[t][ind_l[l]]
            #     il_m[l, l] = il[t][ind_l[l]]
            #     eye_l[ind_l[l], l] = -1.0
            # end

            # for l in 1:nu
            #     zu_m[l, ind_u[l]] = zu[t][ind_u[l]]
            #     iu_m[l, l] = iu[t][ind_u[l]]
            #     eye_u[ind_u[l], l] = 1.0
            # end

            # lhs_full = [(Q̃uu + reg1 .* I(num_control)) transpose(hu[t]) eye_l eye_u;
            #     hu[t] (zeros(num_constr, num_constr) - δ_c1 * I(num_constr)) zeros(num_constr, nl + nu);
            #     zl_m zeros(nl, num_constr) il_m zeros(nl, nu);
            #     -zu_m zeros(nu, num_constr) zeros(nu, nl) iu_m]

            # num_state = length(x[t])
            # rhs_full = -1.0 .* [Q̃u[t] Q̂ux[t];
            #          h[t] hx[t];
            #          (zl[t][ind_l] .* il[t][ind_l] .- μ) zeros(nl, num_state);
            #          (zu[t][ind_u] .* iu[t][ind_u] .- μ) zeros(nu, num_state)]
            
            # display(rhs_full)
            # display(lhs_full)
            # update_rule_ir, bnorm, bcomp = rfldiv(lhs_full, rhs_full; equilibrate=false)
            # update_rule_ir = lhs_full \ rhs_full
    
            # α[t] .= update_rule_ir[1:num_control, 1]
            # β[t] .= update_rule_ir[1:num_control, 2:end]
            # ψ[t] .= update_rule_ir[num_control .+ (1:num_constr), 1]
            # ω[t] .= update_rule_ir[num_control .+ (1:num_constr), 2:end]
            # χl[t][ind_l] .= update_rule_ir[num_control + num_constr .+ (1:nl), 1]
            # ζl[t][ind_l, :] .= update_rule_ir[num_control + num_constr .+ (1:nl), 2:end]
            # χu[t][ind_u] .= update_rule_ir[num_control + num_constr + nl .+ (1:nu), 1]
            # ζu[t][ind_u, :] .= update_rule_ir[num_control + num_constr + nl .+ (1:nu), 2:end]

            # α1 = update_rule_ir[1:num_control, 1]
            # β1 = update_rule_ir[1:num_control, 2:end]
            # ψ1 = update_rule_ir[num_control .+ (1:num_constr), 1]
            # ω1 = update_rule_ir[num_control .+ (1:num_constr), 2:end]
            # χl1 = update_rule_ir[num_control + num_constr .+ (1:nl), 1]
            # ζl1 = update_rule_ir[num_control + num_constr .+ (1:nl), 2:end]
            # χu1 = update_rule_ir[num_control + num_constr + nl .+ (1:nu), 1]
            # ζu1 = update_rule_ir[num_control + num_constr + nl .+ (1:nu), 2:end]

            # update parameters of update rule for ineq. dual variables, i.e., 

            # χ_L =  μ inv.(u - u^L) - z^L - Σ^L * α 
            # ζ_L =  - Σ^L * β 
            # χ_U =  μ inv.(u^U - u) - z^U + Σ^U .* α 
            # ζ_U =  Σ^U .* β

            # see update above for Q̂u[t] for first part of χ^L[t] χ^U[t]

            # println(χl1 - (μ .* inv.(il[t]) - zl[t] - u_tmp1[t] .* α1))
            # println(χu1 - (μ .* inv.(iu[t][ind_u]) - zu[t][ind_u] + u_tmp2[t][ind_u] .* α1[ind_u]))

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

            # println(t, " ", α[t] - α1)
            # println(t, " ", β[t] - β1)
            # println(t, " ", ψ[t] - ψ1)
            # println(t, " ", ω[t] - ω1)
            # println(t, " ", χl[t][ind_l] - χl1)
            # println(t, " ", ζl[t][ind_l, :] - ζl1)
            # println(t, " ", χu[t][ind_u] - χu1)
            # println(t, " ", ζu[t][ind_u, :] - ζu1)

            # throw("arrrrgh")

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


