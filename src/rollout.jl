function rollout!(policy::PolicyData, problem::ProblemData, τ::Float64; step_size=1.0, mode=:main, resto=false)
    dynamics = problem.model.dynamics
    
    x, u, h, il, iu = primal_trajectories(problem, mode=:current)
    ϕ, vl, vu = dual_trajectories(problem, mode=:current)
    p, n, vp, vn = fr_trajectories(problem, mode=:current)
    x̄, ū, h̄, il̄, iū = primal_trajectories(problem, mode=:nominal)
    ϕb, vl̄, vū = dual_trajectories(problem, mode=:nominal)
    p̄, n̄, vp̄, vn̄ = fr_trajectories(problem, mode=:nominal)
    
    x[1] .= x̄[1]

    gains = mode == :main ? policy.gains_main : policy.gains_soc
    ku, kϕ = gains.ku, gains.kϕ
    kvl, kvu = gains.kvl, gains.kvu
    Kvl, Kvu = policy.gains_main.Kvl, policy.gains_main.Kvu
    Ku, Kϕ = policy.gains_main.Ku, policy.gains_main.Kϕ
    kp = policy.kp
    kn = policy.kn
    Kp = policy.Kp
    Kn = policy.Kn
    kvp = policy.kvp
    kvn = policy.kvn
    Kvp = policy.Kvp
    Kvn = policy.Kvn

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

        if resto
            p[k] .= kp[k]
            p[k] .*= step_size
            p[k] .+= p̄[k]
            mul!(p[k], Kp[k], x[k], 1.0, 1.0)
            mul!(p[k], Kp[k], x̄[k], -1.0, 1.0)

            n[k] .= kn[k]
            n[k] .*= step_size
            n[k] .+= n̄[k]
            mul!(n[k], Kn[k], x[k], 1.0, 1.0)
            mul!(n[k], Kn[k], x̄[k], -1.0, 1.0)

            vp[k] .= kvp[k]
            vp[k] .*= step_size
            vp[k] .+= vp̄[k]
            mul!(vp[k], Kvp[k], x[k], 1.0, 1.0)
            mul!(vp[k], Kvp[k], x̄[k], -1.0, 1.0)

            vn[k] .= kvn[k]
            vn[k] .*= step_size
            vn[k] .+= vn̄[k]
            mul!(vn[k], Kvn[k], x[k], 1.0, 1.0)
            mul!(vn[k], Kvn[k], x̄[k], -1.0, 1.0)
        end
        
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
