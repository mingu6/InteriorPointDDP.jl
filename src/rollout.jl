function rollout!(policy::PolicyData{T}, problem::ProblemData{T}; step_size::T=1.0, mode=:main) where T
    dynamics = problem.model.dynamics
    
    x, u, _ = primal_trajectories(problem, mode=:current)
    ϕ, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, _ = primal_trajectories(problem, mode=:nominal)
    ϕ̄, vl̄, vū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    gains = mode == :main ? policy.gains_main : policy.gains_soc
    ku, kϕ = gains.ku, gains.kϕ
    kvl, kvu = gains.kvl, gains.kvu
    Kvl, Kvu = policy.gains_main.Kvl, policy.gains_main.Kvu
    Ku, Kϕ = policy.gains_main.Ku, policy.gains_main.Kϕ

    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        u[t] .= ku[t]
        u[t] .*= step_size
        u[t] .+= ū[t]
        mul!(u[t], Ku[t], x[t], 1.0, 1.0)
        mul!(u[t], Ku[t], x̄[t], -1.0, 1.0)

        # for ϕ, note we use ϕ^+ instead of δϕ for update hence different formula
        ϕ[t] .= kϕ[t]
        ϕ[t] .*= step_size
        ϕ[t] .+= ϕ̄[t]
        mul!(ϕ[t], Kϕ[t], x[t], 1.0, 1.0)
        mul!(ϕ[t], Kϕ[t], x̄[t], -1.0, 1.0)

        vl[t] .= kvl[t]
        vl[t] .*= step_size
        vl[t] .+= vl̄[t]
        mul!(vl[t], Kvl[t], x[t], 1.0, 1.0)
        mul!(vl[t], Kvl[t], x̄[t], -1.0, 1.0)

        vu[t] .= kvu[t]
        vu[t] .*= step_size
        vu[t] .+= vū[t]
        mul!(vu[t], Kvu[t], x[t], 1.0, 1.0)
        mul!(vu[t], Kvu[t], x̄[t], -1.0, 1.0)
        
        dynamics!(d, x[t+1], x[t], u[t])
    end
end
