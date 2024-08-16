function second_order_correction!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options,
                                  τ::Float64)
    data.step_size *= 2.0 # bit dirty, step size should not be decremented before SOC step so this is here
    N = problem.horizon

    x, u, h, il, iu = primal_trajectories(problem, mode=:current)
    ϕ, vl, vu = dual_trajectories(problem, mode=:current)
    x̄, ū, h̄, il̄, iū = primal_trajectories(problem, mode=:nominal)
    ϕb, vl̄, vū = dual_trajectories(problem, mode=:nominal)

    lhs_bk = policy.lhs_bk

    kuϕ = policy.gains_soc.kuϕ
    Ku = policy.gains_main.Ku
    ku = policy.gains_soc.ku
    kvl = policy.gains_soc.kvl
    kvu = policy.gains_soc.kvu
    Kvl = policy.gains_soc.Kvl
    Kvu = policy.gains_soc.Kvu

    α_soc = data.step_size
    θ_soc_old = data.primal_1_curr

    θ_prev = data.primal_1_curr
    φ_prev = data.barrier_obj_curr
    μ = data.μ

    status = false
    p = 1

    for k = N-1:-1:1
        policy.rhs_b[k] .= h̄[k]
    end

    for p = 1:options.n_soc_max
        # backward pass
        for k = N-1:-1:1
            policy.rhs_b[k] .*= α_soc
            policy.rhs_b[k] .+= h[k]

            kuϕ[k] .= lhs_bk[k] \ policy.rhs[k]
            gains_ineq!(kvl[k], kvu[k], Kvl[k], Kvu[k], il̄[k], iū[k], vl̄[k], vū[k], ku[k], Ku[k], μ)
        end
        # forward pass
        α_soc = 1.0  # set high and find max
        while !status && α_soc > eps(Float64)
            try
                rollout!(policy, problem, τ, step_size=α_soc; mode=:soc)
            catch
                # reduces step size if NaN or Inf encountered
                α_soc *= 0.5
                continue
            end
            constraint!(problem, μ; mode=:current)
            status = check_fraction_boundary(problem, τ)
            if status
                break
            else
                α_soc *= 0.5
            end
        end
        !status && (options.verbose && (@warn "SOC iterated failed to pass fraction-to-boundary condition... Weird"))
        !status && break

        # evaluate new iterate against current filter
        θ_soc = constraint_violation_1norm(problem, mode=:current)
        φ_soc = barrier_objective!(problem, data, mode=:current)
        status = !any(x -> all([θ_soc, φ_soc] .>= x), data.filter)
        !status && break

        # evaluate sufficient improvement conditions of SOC iterate
        Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, α_soc)
        Δφ = Δφ_L + Δφ_Q

        data.armijo_passed = φ_soc - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * Δφ
        if (θ_prev <= data.min_primal_1) && data.switching
            status = data.armijo_passed  #  sufficient decrease of barrier objective
        else
            suff = (θ_soc <= (1. - options.γ_θ) * θ_prev) || (φ_soc <= φ_prev - options.γ_φ * θ_prev)
            !suff && (status = false)
        end
        if status
            data.step_size = α_soc
            data.barrier_obj_next = φ_soc
            data.primal_1_next = θ_soc
            data.p = p
            break
        end
        (θ_soc > options.κ_soc * θ_soc_old) && break
        θ_soc_old = θ_soc
    end
    return status
end