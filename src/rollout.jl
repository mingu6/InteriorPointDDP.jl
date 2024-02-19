function rollout!(policy::PolicyData, problem::ProblemData, feasible::Bool; step_size=1.0)
    dynamics = problem.model.dynamics
    constr_data = problem.constraints
    
    x, u, params = primal_trajectories(problem, mode=:current)
    _, s, y = dual_trajectories(constr_data, mode=:current)
    x̄, ū, _ = primal_trajectories(problem, mode=:nominal)
    _, s̄, ȳ = dual_trajectories(constr_data, mode=:nominal)
    
    x[1] .= x̄[1]

    ku, ks, ky = policy.ku, policy.ks, policy.ky  # feedforward gains
    Ku, Ks, Ky = policy.Ku, policy.Ks, policy.Ky  # feedback gains

    for (k, d) in enumerate(dynamics)
        # u[k] .= ū[k] + K[k] * (x[k] - x̄[k]) + step_size * k[k]
        u[k] .= ku[k]
        u[k] .*= step_size
        u[k] .+= ū[k]
        mul!(u[k], Ku[k], x[k], 1.0, 1.0)
        mul!(u[k], Ku[k], x̄[k], -1.0, 1.0)

        # s[k] .= s̄[k] + Ks[k] * (x[k] - x̄[k]) + step_size * ks[k]
        s[k] .= ks[k]
        s[k] .*= step_size
        s[k] .+= s̄[k]
        mul!(s[k], Ks[k], x[k], 1.0, 1.0)
        mul!(s[k], Ks[k], x̄[k], -1.0, 1.0)
        
        if !feasible
            # y[k] .= ȳ[k] + Ky[k] * (x[k] - x̄[k]) + step_size * ky[k]
            y[k] .= ky[k]
            y[k] .*= step_size
            y[k] .+= ȳ[k]
            mul!(y[k], Ky[k], x[k], 1.0, 1.0)
            mul!(y[k], Ky[k], x̄[k], -1.0, 1.0)
        end
        x[k+1] .= dynamics!(d, x[k], u[k], params[k])
    end
end

    
function rollout(dynamics::Vector{Dynamics{T}}, initial_state, actions, 
    parameters=[zeros(d.num_parameter) for d in dynamics]) where T
    x_history = [initial_state]
    for (k, d) in enumerate(dynamics) 
        push!(x_history, copy(dynamics!(d, x_history[end], actions[k], parameters[k])))
    end
    
    return x_history
end
