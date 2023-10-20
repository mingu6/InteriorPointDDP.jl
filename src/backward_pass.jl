using LinearAlgebra

function backward_pass!(policy::PolicyData, 
    solver_data::SolverData,
    problem::ProblemData,
    options::Options,
    mode=:nominal
    )

    # Constraint Data 
    constraint_data = problem.objective.costs.constraint_data
    # Horizon 
    H = length(problem.states)
    # Errors
    dV = [0.0, 0.0]
    c_err::Float64 = 0
    mu_err::Float64 = 0
    Qu_err::Float64 = 0

    # Dimensions 
    dim_x = length(problem.states)
    dim_u = length(problem.actions)
    dim_c = length(constraint_data.constraints)

    # Current trajectory
    x = problem.states
    u = problem.actions

    # Constraints 
    c = constraint_data.constraints

    # Dual variables 
    s = constraint_data.ineq_duals
    
    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    # fxx = problem.model.hessian_state_state
    # fxu = problem.model.hessian_state_action
    # fuu = problem.model.hessian_action_action
    # Jacobian constraints 
    Qsx = constraint_data.jacobian_state
    Qsu = constraint_data.jacobian_action

    # Cost 
    q = problem.objective.costs # objective.evaluate_cache is the cost (V in MATLAB)
    # Cost gradients
    qx = problem.objective.gradient_state
    qu = problem.objective.gradient_action
    # Cost hessians
    qxx = problem.objective.hessian_state_state
    quu = problem.objective.hessian_action_action
    qux = problem.objective.hessian_action_state

    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state

    # Value function
    Vx = policy.value.gradient
    Vxx = policy.value.hessian

    # terminal value function
    # TODO: Extension: Implement terminal constraints 
    Vxx[H] .= qxx[H]
    Vx[H] .= qx[H]

    for t = H-1:-1:1
        # Qx[t] .= qx[t] + (Qsx[t]' * s[t]) + (fx[t]' * Vx[t+1])
        mul!(policy.x_tmp[t], transpose(Qsx[t]), s[t])
        mul!(Qx[t], transpose(fx[t]), Vx[t+1])
        Qx[t] .+= qx[t] + policy.x_tmp[t]

        # Qu[t] .= qu[t] + (Qsu[t]' * s[t]) + (fu[t]' * Vx[t+1])
        mul!(policy.u_tmp[t], transpose(Qsu[t]), s[t])
        mul!(Qu[t], transpose(fu[t]), Vx[t+1])
        Qu[t] .+= qu[t] + policy.u_tmp[t]

        # Qxx[t] .= qxx[t] + ((fx[t]' * Vxx[t+1]) * fx[t])
        mul!(policy.xx̂_tmp[t], transpose(fx[t]), Vxx[t+1])
        mul!(Qxx[t], policy.xx̂_tmp[t], fx[t])
        Qxx[t] .+= qxx[t]

        # Quu[t] .= quu[t] + ((fu[t]' * Vxx[t+1]) * fu[t])
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Quu[t], policy.ux̂_tmp[t], fu[t])
        Quu[t] .+= quu[t]

        # Qux[t] .= qux[t] + ((fu[t]' * Vxx[t+1]) * fx[t])
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Qux[t], policy.ux̂_tmp[t], fx[t])
        Qux[t] .+= qux[t]

        ## TODO: Hacky way since it's symmetric
        Qxu = transpose(Qux[t])
        S = Diagonal(s[t])

        mu = solver_data.perturbation
        Quu_reg = Quu[t] + quu[t] * (1.6^options.reg - 1) 

        # Here is a bunch of future TODO 
        # 1) Factor out ku and Ku 
        # 2) Optimise cholesky decomp
        if options.feasible
            # TODO: Can be recomputed
            # TODO: Call a direct factorisation solver
            r = S * c[t].evaluate_cache .+ mu
            cinv = 1.0 ./ c[t].evaluate_cache
            SCinv = Diagonal(s[t] .* cinv)

            # Compute gains
            compute_gains!(policy, constraint_data, SCinv, cinv, r, S, t, Qxu, Quu_reg)

            # Update action value function 
            update_action_value_function!(policy, constraint_data, SCinv, cinv, r, t)

            # TODO: Move error out once infeasible is complete
            # Error mapping 
            Qu_err = max(Qu_err, norm(Qu, Inf))
            mu_err = max(mu_err, norm(r, Inf))
        end

        update_value_function!(policy, t)

        if !options.feasible
            c_err = max(c_err, norm(fp.c[:,i] + fp.y[:,i], Inf)) # needs changed
        end
    end

    options.opterr=max([Qu_err, c_err, mu_err]);
end


function compute_gains!(policy, constraint_data, SCinv, cinv, r, S, t, Qxu, Quu_reg)

    # Gains 
    Ku = policy.Ku
    ku = policy.ku
    Ks = policy.Ks
    ks = policy.ks
    # Action value gradient w.r.t to action
    Qu = policy.action_value.gradient_action
    # Jacobian constraints 
    Qsx = constraint_data.jacobian_state
    Qsu = constraint_data.jacobian_action

    cholesky_decomp = cholesky(Quu_reg .- Qsu[t]' * SCinv * Qsu[t], check=false)
    
    # TODO: Fix regularisation 
    if !issuccess(cholesky_decomp)
        return false 
    end 

    R = cholesky_decomp.U
    b = hcat(Qu[t] - Qsu[t]' * (cinv .* r), Qxu' - Qsu[t]' * SCinv * Qsx[t])
    kK = -R \ (R' \ b)

    # Update gains for K (Ku) and k (ku)
    ku[t] = kK[:, 1]
    Ku[t] = kK[:, 2:end]
    
    # Update gains for ineq dual variable ks and Ks
    ks[t] = -cinv .* (r .+ S * Qsu[t] * ku[t])
    Ks[t] = -SCinv * (Qsx[t] .+ Qsu[t] * Ku[t])
end


function update_action_value_function!(policy, constraint_data, SCinv, cinv, r, t)
    # Jacobian constraints 
    Qsx = constraint_data.jacobian_state
    Qsu = constraint_data.jacobian_action

    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state

    Quu[t] .-= Qsu[t]' * SCinv * Qsu[t]
    Qux[t] .-= transpose(Qsx[t]' * SCinv * Qsu[t])
    Qxx[t] .-= Qsx[t]' * SCinv * Qsx[t]
    Qu[t] .-= Qsu[t]' * (cinv .* r)
    Qx[t] .-= Qsx[t]' * (cinv .* r)
end

function update_value_function!(policy, t)

    # Value function approximations
    Vx = policy.value.gradient
    Vxx = policy.value.hessian
    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state
    # Gains values
    Ku = policy.Ku
    ku = policy.ku

    # Vxx[t] .=  Qxx[t] + (K[t]' * (Quu[t] * K[t])) + (K[t]' * Qux[t]) + (Qux[t]' * K[t])
    mul!(policy.ux_tmp[t], Quu[t], Ku[t])
    mul!(Vxx[t], transpose(Ku[t]), policy.ux_tmp[t])
    mul!(Vxx[t], transpose(Ku[t]), Qux[t], 1.0, 1.0) # apply appropriate scaling 
    mul!(Vxx[t], transpose(Qux[t]), Ku[t], 1.0, 1.0) # apply appropriate scaling
    Vxx[t] .+= Qxx[t]

    # Vx[t] .=  Qx[t] + (K[t]' * Quu[t] * k[t]) + (K[t]' * Qu[t]) + (Qux[t]' * k[t])
    mul!(policy.u_tmp[t], Quu[t], ku[t])
    mul!(Vx[t], transpose(Ku[t]), policy.u_tmp[t])
    mul!(Vx[t], transpose(Ku[t]), Qu[t], 1.0, 1.0) # apply appropriate scaling 
    mul!(Vx[t], transpose(Qux[t]), ku[t], 1.0, 1.0) # apply appropriate scaling 
    Vx[t] .+= Qx[t]

end