function rollout!(policy::PolicyData, problem::ProblemData, feasible::Bool; step_size=1.0)
    dynamics = problem.model.dynamics

    # trajectories
    x = problem.states
    u = problem.actions
    params = problem.parameters
    constr_data = problem.constraints
    s = constr_data.ineq_duals
    y = constr_data.slacks
    x̄ = problem.nominal_states
    ū = problem.nominal_actions
    s̄ = constr_data.nominal_ineq_duals
    ȳ = constr_data.nominal_slacks
    x[1] .= x̄[1]

    # policy
    Ku = policy.Ku
    ku = policy.ku
    Ks = policy.Ks
    ks = policy.ks
    Ky = policy.Ky
    ky = policy.ky

    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        u[t] .= ku[t]
        u[t] .*= step_size
        u[t] .+= ū[t]
        mul!(u[t], Ku[t], x[t], 1.0, 1.0)
        mul!(u[t], Ku[t], x̄[t], -1.0, 1.0)

        # s[t] .= s̄[t] + Ks[t] * (x[t] - x̄[t]) + step_size * ks[t]
        s[t] .= ks[t]
        s[t] .*= step_size
        s[t] .+= s̄[t]
        mul!(s[t], Ks[t], x[t], 1.0, 1.0)
        mul!(s[t], Ks[t], x̄[t], -1.0, 1.0)
        
        if !feasible
            # y[t] .= ȳ[t] + Ky[t] * (x[t] - x̄[t]) + step_size * ky[t]
            y[t] .= ky[t]
            y[t] .*= step_size
            y[t] .+= ȳ[t]
            mul!(y[t], Ky[t], x[t], 1.0, 1.0)
            mul!(y[t], Ky[t], x̄[t], -1.0, 1.0)
        end
        x[t+1] .= dynamics!(d, x[t], u[t], params[t])
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
