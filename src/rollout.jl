function rollout!(policy::PolicyData, problem::ProblemData, τ::Float64; step_size=1.0, mode=:main)
    dynamics = problem.model.dynamics
    
    x, u, _ = primal_trajectories(problem, mode=:current)
    ϕ, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, _ = primal_trajectories(problem, mode=:nominal)
    ϕb, vl̄, vū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    gains = mode == :main ? policy.gains_main : policy.gains_soc
    ku, kϕ = gains.ku, gains.kϕ
    kvl, kvu = gains.kvl, gains.kvu
    Kvl, Kvu = policy.gains_main.Kvl, policy.gains_main.Kvu
    Ku, Kϕ = policy.gains_main.Ku, policy.gains_main.Kϕ

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
        ϕ[k] .+= ϕb[k]
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
        
        dynamics!(d, x[k+1], x[k], u[k])
    end
end
