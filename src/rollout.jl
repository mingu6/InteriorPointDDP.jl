function rollout!(policy::PolicyData, problem::ProblemData, τ::Float64; step_size=1.0, mode=:main)
    dynamics = problem.model.dynamics
    
    x, u, h, il, iu = primal_trajectories(problem, mode=:current)
    ϕ, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, h̄, il̄, iū = primal_trajectories(problem, mode=:nominal)
    ϕb, vl̄, vū = dual_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    gains = mode == :main ? policy.gains_main : policy.gains_soc
    ku, kϕ = gains.ku, gains.kϕ
    kvl, kvu = gains.kvl, gains.kvu
    Kvl, Kvu = policy.gains_main.Kvl, policy.gains_main.Kvu
    Ku, Kϕ = policy.gains_main.Ku, policy.gains_main.Kϕ

    step_dual = 1.0

    N = length(dynamics) + 1

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
        # ϕ[k] .+= (1. - step_size) * ϕb[k]
        ϕ[k] .+= ϕb[k]
        mul!(ϕ[k], Kϕ[k], x[k], 1.0, 1.0)
        mul!(ϕ[k], Kϕ[k], x̄[k], -1.0, 1.0)
        
        # # take independent steps for duals using max step length
        # step_vl = max_step_dual(vl̄[k], kvl[k], τ)
        # step_vu = max_step_dual(vū[k], kvu[k], τ)
        # step_dual = min(step_dual, step_vl, step_vu)

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

    # step_dual = max(step_dual - 1e-6, 0.0) 

    # for k = 1:N-1
    #     vl[k] .= kvl[k]
    #     vl[k] .*= step_dual
    #     vl[k] .+= vl̄[k]

    #     vu[k] .= kvu[k]
    #     vu[k] .*= step_dual
    #     vu[k] .+= vū[k]
    # end
end

    
function rollout(dynamics::Vector{Dynamics{T}}, initial_state, actions) where T
    x_history = [initial_state]
    for (k, d) in enumerate(dynamics) 
        push!(x_history, copy(dynamics!(d, x_history[end], actions[k])))
    end
    
    return x_history
end

function max_step_dual(v::Vector{T}, kv::Vector{T}, τ::T) where T
    n = length(v)
    step_size = 1.0
    for i = 1:n
        if iszero(kv[i])
            continue
        else
            a = -τ * v[i] / kv[i]
            step_size = a < 0 ? min(step_size, 1.0) : min(step_size, min(a, 1.0))
        end
    end
    return step_size
end
