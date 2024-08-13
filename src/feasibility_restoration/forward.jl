function fr_forward_pass!(policy::PolicyData, problem::ProblemData, data::SolverData, options::Options,
                       u_R::Vector{T}, D_R::Vector{T}, ρ::Float64, ζ::Float64; verbose=false) where T
    data.l_R = 0  # line search iteration counter
    data.status = true
    data.step_size = 1.0
    Δφ = 0.0
    μ = data.μ
    τ = max(options.τ_min, 1.0 - μ)

    θ_prev = data.primal_1_curr
    φ_prev = data.barrier_obj_curr
    θ = θ_prev

    Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, data.step_size)
    Δφ = Δφ_L + Δφ_Q
    min_step_size = estimate_min_step_size(Δφ_L, data, options)

    while data.step_size >= min_step_size
        α = data.step_size
        
        try
            rollout!(policy, problem, τ, step_size=α; mode=:main, resto=true)
        catch
            # reduces step size if NaN or Inf encountered
            data.step_size *= 0.5
            continue
        end

        fr_constraint!(problem, mode=:current)
        
        data.status = check_fraction_boundary(problem, τ; resto=true)
        # println("boundary failed")
        !data.status && (data.step_size *= 0.5, continue)

        Δφ_L, Δφ_Q = expected_decrease_cost(policy, problem, α; mode=:main)
        Δφ = Δφ_L + Δφ_Q
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = constraint_violation_1norm(problem, mode=:current)
        φ = fr_barrier_objective!(problem, u_R, D_R, μ, ρ, ζ, mode=:current)
        
        # check acceptability to filter
        data.status = !any(x -> all([θ, φ] .>= x), data.filter)
        # println("filter ", data.k, " ", data.status, " ", α, " ", θ, " ", φ)
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        # check for sufficient decrease conditions for the barrier objective/constraint violation
        data.switching = (Δφ < 0.0) && 
            ((-Δφ) ^ options.s_φ * α^(1-options.s_φ)  > options.δ * θ_prev ^ options.s_θ)
        # println("switch ", data.switching, " ", θ_prev, " ", -Δφ, " ", min_step_size)
        data.armijo_passed = φ - φ_prev - 10. * eps(Float64) * abs(φ_prev) <= options.η_φ * Δφ
        # println("armijo ", data.armijo_passed, " ", θ <= data.min_primal_1, " ", φ - φ_prev, " ", Δφ)
        if (θ <= data.min_primal_1) && data.switching
            data.status = data.armijo_passed  #  sufficient decrease of barrier objective
        else
            suff = (θ <= (1. - options.γ_θ) * θ_prev) || (φ <= φ_prev - options.γ_φ * θ_prev)
            # println("suff ", suff, " ", θ, " ", (1. - options.γ_θ) * θ_prev, " ", φ, " ", φ_prev - options.γ_φ * θ_prev)
            !suff && (data.status = false)
        end
        !data.status && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        data.barrier_obj_next = φ
        data.primal_1_next = θ
        break
    end
    # println("forward sucecss ", data.step_size)
    data.step_size < min_step_size && (data.status = false)
    !data.status && (verbose && (@warn "(FR) Line search failed to find a suitable iterate"))
end