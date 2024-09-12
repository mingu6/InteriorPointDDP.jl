function rollout!(policy::PolicyData{T}, problem::ProblemData{T}; step_size::T=1.0) where T
    dynamics = problem.model.dynamics
    
    x, u, _ = primal_trajectories(problem, mode=:current)
    ϕ, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, _ = primal_trajectories(problem, mode=:nominal)
    ϕ̄, vl̄, vū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    α, ψ = policy.gains_data.α, policy.gains_data.ψ
    β, ω = policy.gains_data.β, policy.gains_data.ω
    χl, χu = policy.gains_data.χl, policy.gains_data.χu
    ζl, ζu = policy.gains_data.ζl, policy.gains_data.ζu

    for (t, d) in enumerate(dynamics)
        # u[t] .= ū[t] + K[t] * (x[t] - x̄[t]) + step_size * k[t]
        u[t] .= α[t]
        u[t] .*= step_size
        u[t] .+= ū[t]
        mul!(u[t], β[t], x[t], 1.0, 1.0)
        mul!(u[t], β[t], x̄[t], -1.0, 1.0)

        # for ϕ, note we use ϕ^+ instead of δϕ for update hence different formula
        ϕ[t] .= ψ[t]
        ϕ[t] .*= step_size
        ϕ[t] .+= ϕ̄[t]
        mul!(ϕ[t], ω[t], x[t], 1.0, 1.0)
        mul!(ϕ[t], ω[t], x̄[t], -1.0, 1.0)

        vl[t] .= χl[t]
        vl[t] .*= step_size
        vl[t] .+= vl̄[t]
        mul!(vl[t], ζl[t], x[t], 1.0, 1.0)
        mul!(vl[t], ζl[t], x̄[t], -1.0, 1.0)

        vu[t] .= χu[t]
        vu[t] .*= step_size
        vu[t] .+= vū[t]
        mul!(vu[t], ζu[t], x[t], 1.0, 1.0)
        mul!(vu[t], ζu[t], x̄[t], -1.0, 1.0)
        
        dynamics!(d, x[t+1], x[t], u[t])
    end
end
