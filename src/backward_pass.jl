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
        
        for t = N-1:-1:1
            num_control = length(u[t])

            bk = bounds[t]
            il = u[t][bk.indices_lower] - bk.lower[bk.indices_lower]
            iu = bk.upper[bk.indices_upper] - u[t][bk.indices_upper]
            σl = vl[t][bk.indices_lower] ./ il
            σu = vu[t][bk.indices_upper] ./ iu

            # Qx = lx + fx' * Vx
            Qx[t] .= lx[t]
            mul!(Qx[t], transpose(fx[t]), Vx[t+1], 1.0, 1.0)

            # Qu = lu + fu' * Vx + μ log(u - ul) + μ log(uu - u)
            Qu[t] .= lu[t]
            mul!(Qu[t], transpose(fu[t]), Vx[t+1], 1.0, 1.0)
            Qu[t][bk.indices_lower] .-= μ ./ il
            Qu[t][bk.indices_upper] .+= μ ./ iu
            
            # Qxx = lxx + fx' * Vxx * fx
            mul!(policy.xx_tmp[t], transpose(fx[t]), Vxx[t+1])
            mul!(Qxx[t], policy.xx_tmp[t], fx[t])
            Qxx[t] .+= lxx[t]
    
            # Quu = luu + fu' * Vxx * fu + Σ
            mul!(policy.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
            mul!(Quu[t], policy.ux_tmp[t], fu[t])
            Quu[t] .+= luu[t]
            Quu[t][diagind(Quu[t])[bk.indices_lower]] .+= σl
            Quu[t][diagind(Quu[t])[bk.indices_upper]] .+= σu
    
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
            # setup linear system in backward pass
            policy.lhs_tl[t] .= Quu[t]
            policy.lhs_tr[t] .= transpose(hu[t])
            fill!(policy.lhs_br[t], 0.0)
            if !options.quasi_newton
                policy.lhs_tl[t] .+= vhuu[t]
            end
            policy.lhs[t] .= Symmetric(policy.lhs[t])

            policy.rhs_t[t] .= -Qu[t]
            mul!(policy.rhs_t[t], transpose(hu[t]), ϕ[t], -1.0, 1.0)
            policy.rhs_b[t] .= -h[t]

            policy.rhs_x_t[t] .= -Qux[t]
            policy.rhs_x_b[t] .= -hx[t]
            if !options.quasi_newton
                policy.rhs_x_t[t] .-= vhux[t]
            end

            # inertia calculation and correction
            policy.lhs_tl[t][diagind(policy.lhs_tl[t])] .+= reg
            policy.lhs_br[t][diagind(policy.lhs_br[t])] .-= δ_c

            policy.lhs_bk[t], data.status, reg, δ_c = inertia_correction!(policy.lhs[t], num_control,
                        μ, reg, data.reg_last, options; rook=true)

            data.status != 0 && break

            kuϕ[t] .= policy.lhs_bk[t] \ policy.rhs[t]
            Kuϕ[t] .= policy.lhs_bk[t] \ policy.rhs_x[t]

            # update gains for ineq. dual variables 
            # TODO: improve, copying and slow
            kvl[t][bk.indices_lower] .= μ ./ il - vl[t][bk.indices_lower] - σl .* ku[t][bk.indices_lower]
            kvu[t][bk.indices_upper] .= μ ./ iu - vu[t][bk.indices_upper] + σu .* ku[t][bk.indices_upper]
            Kvl[t][bk.indices_lower, :] .= -σl .* Ku[t][bk.indices_lower, :]
            Kvu[t][bk.indices_upper, :] .= σu .* Ku[t][bk.indices_upper, :]

            # Update return function approx. for next timestep 
            # Vxx = Q̂xx + Q̂ux' * Ku + Ku * Q̂ux' + Ku' Q̂uu' * Ku
            mul!(policy.ux_tmp[t], Quu[t], Ku[t])
            mul!(Vxx[t], transpose(Ku[t]), Qux[t])

            mul!(Vxx[t], transpose(Qux[t]), Ku[t], 1.0, 1.0)
            Vxx[t] .+= Qxx[t]
            mul!(Vxx[t], transpose(Ku[t]), policy.ux_tmp[t], 1.0, 1.0)
            Vxx[t] .= Symmetric(Vxx[t])

            # Vx = Q̂x + Ku' * Q̂u + [Q̂uu Ku + Q̂ux]^T ku
            policy.ux_tmp[t] .+= Qux[t]
            mul!(Vx[t], transpose(policy.ux_tmp[t]), ku[t])
            mul!(Vx[t], transpose(Ku[t]), Qu[t], 1.0, 1.0)
            Vx[t] .+= Qx[t]
        end
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find positive definite iteration matrix."))
end


