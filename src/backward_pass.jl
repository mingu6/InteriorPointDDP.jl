function backward_pass!(policy::PolicyData, 
    problem::ProblemData; 
    mode=:nominal, 
    horizon::Int128, 
    constraint_data::ConstraintData)

    # Horizon 
    H = horizon

    # Dimensions 
    dim_x = length(problem.states)
    dim_u = length(problem.actions)
    dim_c = length(constraint_data.constraints)

    # Errors
    c_err::Float64 = 0
    mu_err::Float64 = 0
    Qu_err::Float64 = 0

    # Current trajectory
    x = problem.states
    u = problem.actions

    # Dual variables 
    # TODO: Interface with dual variables properly
    s = problem.dual.eq
    y = problem.dual.ineq
    
    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_action
    # Hessians of system dynamics 
    fxx = problem.model.hessian_state_state
    fxu = problem.model.hessian_state_action
    fuu = problem.model.hessian_action_action
    # Jacobian constraints 
    Qsx = constraint_data.jacobian_state
    Qsu = constraint_data.jacobian_action

    # Cost 
    q = problem.objective.costs
    # Cost gradients
    qx = problem.objective.gradient_state
    qu = problem.objective.gradient_action
    # Cost hessians
    qxx = problem.objective.hessian_state_state
    quu = problem.objective.hessian_action_action
    qux = problem.objective.hessian_action_state

    # Value function approximation
    V = policy.value
    Vx = policy.value.gradient
    Vxx = policy.value.hessian

    # Action-Value function approximation
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    Qxx = policy.action_value.hessian_state_state
    Quu = policy.action_value.hessian_action_action
    Qux = policy.action_value.hessian_action_state

    # policy
    # TODO: Update policy to include dual variables
    if mode == :nominal
        K = policy.K
        k = policy.k
    else
        K = policy.K_candidate
        k = policy.k_cand
    end

    # terminal value function
    # TODO: Update terminal value function - is this correct?
    # TODO: Implement without possibility of terminal constraints - this is an extension 
    Vxx[H] .= qxx[H]
    Vx[H] .= qx[H]

    for t = H-1:-1:1
        # TODO: Question - Why do we multiply the jacobian state by t+1th timestep of the value function?

        # Qx[t] .= qx[t] + (Qsx[t]' * s)+ (fx[t]' * Vx[t+1])
        # TODO: Need to pre-allocate the matrices here in the policy struct and then add
        # TODO: Optimise tensordot and tensor-vector product
        mul!(t1, transpose(fx[t]), Vx[t+1])
        mul!(t2, transpose(Qsx[t]), s)
        Qx[t] .+= qx[t] + t1 + t2

        # Qu[t] .= gu[t] + fu[t]' * p[t+1]
        mul!(Qu[t], transpose(fu[t]), Vx[t+1])
        Qu[t] .+= qu[t]

        # Qxx[t] .= gxx[t] + fx[t]' * P[t+1] * fx[t]
        mul!(policy.xx̂_tmp[t], transpose(fx[t]), Vxx[t+1])
        mul!(Qxx[t], policy.xx̂_tmp[t], fx[t])
        Qxx[t] .+= qxx[t]

        # Quu[t] .= guu[t] + fu[t]' * P[t+1] * fu[t]
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Quu[t], policy.ux̂_tmp[t], fu[t])
        Quu[t] .+= quu[t]

        # Qux[t] .= gux[t] + fu[t]' * P[t+1] * fx[t]
        mul!(policy.ux̂_tmp[t], transpose(fu[t]), Vxx[t+1])
        mul!(Qux[t], policy.ux̂_tmp[t], fx[t])
        Qux[t] .+= qux[t]

        # K[t] .= -1.0 * Quu[t] \ Qux[t]
        # k[t] .= -1.0 * Quu[t] \ Qu[t]
		policy.uu_tmp[t] .= Quu[t]
        LAPACK.potrf!('U', policy.uu_tmp[t])
        K[t] .= Qux[t]
        k[t] .= Qu[t]
        LAPACK.potrs!('U', policy.uu_tmp[t], K[t])
		LAPACK.potrs!('U', policy.uu_tmp[t], k[t])
		K[t] .*= -1.0
		k[t] .*= -1.0

        # P[t] .=  Qxx[t] + K[t]' * Quu[t] * K[t] + K[t]' * Qux[t] + Qux[t]' * K[t]
        # p[t] .=  Qx[t] + K[t]' * Quu[t] * k[t] + K[t]' * Qu[t] + Qux[t]' * k[t]
		mul!(policy.ux_tmp[t], Quu[t], K[t])

		mul!(Vxx[t], transpose(K[t]), policy.ux_tmp[t])
		mul!(Vxx[t], transpose(K[t]), Qux[t], 1.0, 1.0)
		mul!(Vxx[t], transpose(Qux[t]), K[t], 1.0, 1.0)
		Vxx[t] .+= Qxx[t]

		mul!(Vx[t], transpose(policy.ux_tmp[t]), k[t])
		mul!(Vx[t], transpose(K[t]), Qu[t], 1.0, 1.0)
		mul!(Vx[t], transpose(Qux[t]), k[t], 1.0, 1.0)
		Vx[t] .+= Qx[t]
    end
end
