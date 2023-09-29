function rollout!(policy::PolicyData, problem::ProblemData, constraints::ConstraintsData; 
    step_size=1.0)

    # model 
    dynamics = problem.model.dynamics

    # trajectories
    x = problem.states
    u = problem.actions
    w = problem.parameters
    x̄ = problem.nominal_states # old states
    ū = problem.nominal_actions # old actions

    # TODO: need to grab s and s̄ somehow??!?
    s = constraints.ineq
    s̄ = constraints.nominal_ineq # old dual
    
    # policy
    K = policy.K
    k = policy.k

    Ks = policy.Ks
    ks = policy.ks

    # initial state
    x[1] .= x̄[1]

    τ = max(0.99, 1 - constraints.μ)
    is_satisfied = true

    # rollout
    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        u[t] .= k[t] 
        u[t] .*= step_size 
        u[t] .+= ū[t] 
        mul!(u[t], K[t], x[t], 1.0, 1.0) 
        mul!(u[t], K[t], x̄[t], -1.0, 1.0)

        # s[t] .= s̄[t] + Ks[t] * (x[t] -x̄[t]) + step_size * ks[t]
        # s_temp = s[t]
        s[t] .*= step_size
        s[t] .+= s̄[t]
        mul!(s[t], K[t], x[t], 1.0, 1.0)  # s[t] = s[t] + step_size * ks[t] + Ks[t] * x[t] 
        mul!(s[t], K[t], x̄[t], -1.0, 1.0) # s[t] = s[t] + step_size * ks[t] + Ks[t] * x[t] - Ks[t] * x̄[t]


        # TODO: check this is correct
        # check strict constraint satisfaction
        for (i, dual_value) in enumerate(s[t])
            if dual_value <  (1 - τ) * s̄[t][i]
                is_satisfied = false
                return is_satisfied
            end
        end

        # check dual variable positivity
        con = constraints.constraints[t]
        con_temp = constraints.constraints[t]
        con.num_constraint == 0 && continue
        con.evaluate(con.evaluate_cache, states[t], actions[t], parameters[t])
        for index in con.indices_inequality
            if con.evaluate_cache[index] > (1 - τ) * con_temp.violations[index] # calculations are stored in violations
                is_satisfied = false
                return is_satisfied
            end
        end

        x[t+1] .= dynamics!(d, x[t], u[t], w[t])
        return is_satisfied
    end
end

function rollout(dynamics::Vector{Dynamics{T}}, initial_state, actions, 
    parameters=[zeros(d.num_parameter) for d in dynamics]) where T

    x_history = [initial_state]
    for (t, d) in enumerate(dynamics) 
        push!(x_history, copy(dynamics!(d, x_history[end], actions[t], parameters[t])))
    end
    
    return x_history
end
