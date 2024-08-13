function feasibility_restoration!(problem, policy, data, options)
    μ_R = max(data.μ, data.primal_inf)
    ζ = sqrt(μ_R)
    ρ = options.ρ
    data.FR = true
    data.j_R = 0
    data.k_R = 0

    reg_last = data.reg_last
    μ_std = data.μ
    data.μ = μ_R
    vl_old = deepcopy(problem.nominal_ineq_duals_lo)
    vu_old = deepcopy(problem.nominal_ineq_duals_up)

    reset_duals!(problem)

    actions_ref = deepcopy(problem.nominal_actions)
    θ_R = data.primal_1_curr  # reference point constraint violation for termination
    old_filter = deepcopy(data.filter)

    # scaling factor for penalty to nominal trajectory
    D_R = deepcopy(actions_ref)
    f_dr = x -> min.(1.0, 1.0 ./ abs.(x))
    D_R = map(f_dr, D_R)

    # init dual variables
    f_z = z -> min.(ρ, z)
    problem.nominal_ineq_duals_lo = map(f_z, problem.nominal_ineq_duals_lo)
    problem.nominal_ineq_duals_up = map(f_z, problem.nominal_ineq_duals_up)

    # init slack variables for restoration phase
    constraint!(problem; mode=:nominal)
    f_n = c -> (μ_R .- ρ * c) ./ (2.0 .* ρ) .+ sqrt.(((μ_R .- ρ .* c) ./ (2.0 .* ρ)).^2 .+ μ_R .* c ./ (2.0 .* ρ)) 
    problem.nominal_n = map(f_n, problem.nominal_constraints)
    problem.nominal_p = problem.nominal_n + problem.nominal_constraints

    # init dual slack variables for restoration phase
    f_vp = p -> μ_R  ./ p
    problem.nominal_vp = map(f_vp, problem.nominal_p)
    problem.nominal_vn = map(f_vp, problem.nominal_n)

    # update performance measures for first iterate (req. for sufficient decrease conditions for step acceptance)
    fr_constraint!(problem; mode=:nominal)
    data.primal_1_curr = constraint_violation_1norm(problem, mode=:nominal)
    data.barrier_obj_curr = fr_barrier_objective!(problem, actions_ref, D_R, μ_R, ρ, ζ, mode=:nominal)
    
    # filter initialization for constraint violation and threshold for switching rule init. (step acceptance)
    data.max_primal_1 = 1e4 * max(1.0, data.primal_1_curr)
    data.min_primal_1 = 1e-4 * max(1.0, data.primal_1_curr)
    reset_filter!(data)

    while data.k_R < options.max_iterations
        iter_time = @elapsed begin
            fr_constraint!(problem; mode=:nominal)
            gradients!(problem, mode=:nominal)
            fr_backward_pass!(policy, problem, data, options, actions_ref, D_R, data.μ, ρ, ζ; mode=:nominal, verbose=options.verbose)
            if !data.status
                status = 2
                break
            end
            # check (outer) overall problem convergence
            data.dual_inf, data.primal_inf, data.cs_inf = optimality_error(policy, problem, data, options, 0.0; mode=:nominal)
            opt_err_0 = max(data.dual_inf, data.cs_inf, data.primal_inf)

            # if FR phase sufficiently decreases constraint violation and is acceptable to filter, terminate 
            constraint!(problem, mode=:nominal)  # eval constraint w/out slack variables to measure improvement
            θ_curr = constraint_violation_1norm(problem, mode=:nominal)
            # println("actual ", θ_curr)
            φ_curr = barrier_objective!(problem, data, mode=:nominal)

            # termination criteria for FR phase
            FR_success = θ_curr < options.κ_resto * θ_R && !any(x -> all([θ_curr, φ_curr] .>= x), old_filter)
            if FR_success
                println("FR success ", θ_curr)
                data.status = true
                break
            end
            if opt_err_0 <= options.optimality_tolerance
                data.status = false
                options.verbose && (@warn "Solver failed, found a local minimiser of constraint violation.")
                break
            end

            # check (inner) barrier problem convergence and update barrier parameter if so
            fr_constraint!(problem; mode=:nominal)
            dual_inf_μ, primal_inf_μ, cs_inf_μ = optimality_error(policy, problem, data, options, data.μ; mode=:nominal)
            opt_err_μ = max(dual_inf_μ, cs_inf_μ, primal_inf_μ)          
            if opt_err_μ <= options.κ_ϵ * data.μ
                data.μ = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ, data.μ ^ options.θ_μ))
                reset_filter!(data)
                # performance of current iterate updated to account for barrier parameter change
                data.barrier_obj_curr = fr_barrier_objective!(problem, actions_ref, D_R, data.μ, ρ, ζ; mode=:nominal)
                data.j_R += 1
                ζ = sqrt(data.μ)
                continue
            end
            fr_forward_pass!(policy, problem, data, options, actions_ref, D_R, ρ, ζ; verbose=options.verbose)
            if !data.status
                problem.nominal_n = map(f_n, problem.nominal_constraints)
                problem.nominal_p = problem.nominal_n + problem.nominal_constraints
                problem.nominal_vp = map(f_vp, problem.nominal_p)
                problem.nominal_vn = map(f_vp, problem.nominal_n)
                break
            end
            
            rescale_duals!(problem, data.μ, options)  # TODO: add slack variables and duals to this
            update_nominal_trajectory!(problem; resto=true)
            (!data.armijo_passed || !data.switching) && update_filter!(data, options)
            data.barrier_obj_curr = data.barrier_obj_next
            data.primal_1_curr = data.primal_1_next
        end
        
        data.k_R += 1
        data.wall_time += iter_time
    end
    if data.k_R == options.max_iterations
        data.status = false
        options.verbose && (@warn "Feasibility restoration phase failed, max. iterations reached.")
    end
    data.filter = old_filter
    # reset_filter!(data)
    # TODO: if success, set the dual variables ineq of the iterate
    for k = 1:problem.horizon-1
        fill!(problem.nominal_eq_duals[k], 0.0)
    end
    data.μ = μ_std
    data.reg_last = reg_last
    # TODO: resto of resto
    problem.nominal_ineq_duals_lo = vl_old
    problem.nominal_ineq_duals_up = vu_old
end

function fr_constraint!(problem; mode=:nominal)
    N = problem.horizon
    constraint!(problem; mode)
    h = mode == :nominal ? problem.nominal_constraints : problem.constraints
    p = mode == :nominal ? problem.nominal_p : problem.p
    n = mode == :nominal ? problem.nominal_n : problem.n
    for k = 1:N-1
        h[k] .-= p[k]
        h[k] .+= n[k]
    end
end

function fr_barrier_objective!(problem, actions_ref, D_R, μ, ρ, ζ; mode=:nominal)
    N = problem.horizon
    constr_data = problem.constr_data
    _, u, _, il, iu = primal_trajectories(problem, mode=mode)
    p = mode == :nominal ? problem.nominal_p : problem.p
    n = mode == :nominal ? problem.nominal_n : problem.n
    
    barrier_obj = 0.
    for k = 1:N-1
        constr = constr_data.constraints[k]
        for i = 1:constr.num_action
            if !isinf(il[k][i])
                barrier_obj -= μ * log(il[k][i])
            end
            if !isinf(iu[k][i])
                barrier_obj -= μ * log(iu[k][i])
            end
        end
        barrier_obj += ρ * sum(p[k])
        barrier_obj += ρ * sum(n[k])
        barrier_obj += 0.5 * ζ * sum((D_R[k] .* (u[k] - actions_ref[k])).^2)
        barrier_obj -= μ * sum(log.(p[k]))
        barrier_obj -= μ * sum(log.(n[k]))
    end
    return barrier_obj
end
