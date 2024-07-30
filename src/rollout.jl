function rollout!(policy::PolicyData, problem::ProblemData; step_size=1.0)
    dynamics = problem.model.dynamics
    
    x, u, h, il, iu = primal_trajectories(problem, mode=:current)
    ϕ, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, h̄, il̄, iū = primal_trajectories(problem, mode=:nominal)
    ϕb, vl̄, vū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    ku, kϕ, kvl, kvu = policy.ku, policy.kϕ, policy.kvl, policy.kvu
    Ku, Kϕ, Kvl, Kvu = policy.Ku, policy.Kϕ, policy.Kvl, policy.Kvu

    for (k, d) in enumerate(dynamics)
        # u[k] .= ū[k] + K[k] * (x[k] - x̄[k]) + step_size * k[k]
        u[k] .= ku[k]
        u[k] .*= step_size
        u[k] .+= ū[k]
        mul!(u[k], Ku[k], x[k], 1.0, 1.0)
        mul!(u[k], Ku[k], x̄[k], -1.0, 1.0)

        # for ϕ, note we use ϕ^+ instead of δϕ for update hence different formula
        ϕ[k] .= kϕ[k]
        ϕ[k] .*= step_size
        ϕ[k] .+= (1. - step_size) * ϕb[k]
        mul!(ϕ[k], Kϕ[k], x[k], 1.0, 1.0)
        mul!(ϕ[k], Kϕ[k], x̄[k], -1.0, 1.0)

        vl[k] .= kvl[k]
        vl[k] .*= step_size
        vl[k] .+= vl̄[k]
        mul!(vl[k], Kvl[k], x[k], 1.0, 1.0)
        mul!(vl[k], Kvl[k], x̄[k], -1.0, 1.0)

        vu[k] .= kvu[k]
        vu[k] .*= step_size
        vu[k] .+= vū[k]
        mul!(vu[k], Kvu[k], x[k], 1.0, 1.0)
        mul!(vu[k], Kvu[k], x̄[k], -1.0, 1.0)
        
        x[k+1] .= dynamics!(d, x[k], u[k])
    end
end

    
function rollout(dynamics::Vector{Dynamics{T}}, initial_state, actions) where T
    x_history = [initial_state]
    for (k, d) in enumerate(dynamics) 
        push!(x_history, copy(dynamics!(d, x_history[end], actions[k])))
    end
    
    return x_history
end
