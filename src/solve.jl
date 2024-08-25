# function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Costs{T}}
#     ipddp_solve!(solver, args...; kwargs...)
# end

function solve!(solver::Solver{T}, args...; kwargs...) where T
    ipddp_solve!(solver, args...; kwargs...)
end

# TODO: remove ipddp_solve and just have solve!

function ipddp_solve!(solver::Solver, x1, actions; kwargs...)
    initialize_trajectory!(solver, actions, x1)
    ipddp_solve!(solver; kwargs...)
end

function ipddp_solve!(solver::Solver)
    (solver.options.verbose && solver.data.k==0) && solver_info()
	
	policy = solver.policy
    problem = solver.problem
    options = solver.options
	data = solver.data
    
    reset!(problem.model)
    reset!(problem.cost_data)
    reset!(data)
    reset_duals!(problem)  # TODO: initialize better, wrap up with initialization of problem data
    
    # automatically select initial perturbation. loosely based on bound of CS condition (duality) for LPs
    cost!(data, problem, mode=:nominal)
    # data.μ = (data.μ == 0.0) ? options.μ_init * data.objective / max(problem.constr_data.num_constraints[1], 1.0) : data.μ
    data.μ = 1.0

    constraint!(problem, data.μ; mode=:nominal)
    
    # update performance measures for first iterate (req. for sufficient decrease conditions for step acceptance)
    data.primal_1_curr = constraint_violation_1norm(problem, mode=:nominal)
    data.barrier_obj_curr = barrier_objective!(problem, data, mode=:nominal)
    
    # filter initialization for constraint violation and threshold for switching rule init. (step acceptance)
    data.max_primal_1 = 1e4 * max(1.0, data.primal_1_curr)
    data.min_primal_1 = 1e-4 * max(1.0, data.primal_1_curr)
    reset_filter!(data)

    while data.k < options.max_iterations
        iter_time = @elapsed begin
            gradients!(problem, mode=:nominal)
            
            backward_pass!(policy, problem, data, options, mode=:nominal, verbose=options.verbose)
            data.status != 0 && break
            # check (outer) overall problem convergence

            data.dual_inf, data.primal_inf, data.cs_inf = optimality_error(policy, problem, data, options, 0.0, mode=:nominal)
            opt_err_0 = max(data.dual_inf, data.cs_inf, data.primal_inf)
            
            opt_err_0 <= options.optimality_tolerance && break
            
            # check (inner) barrier problem convergence and update barrier parameter if so
            dual_inf_μ, primal_inf_μ, cs_inf_μ = optimality_error(policy, problem, data, options, data.μ, mode=:nominal)
            opt_err_μ = max(dual_inf_μ, cs_inf_μ, primal_inf_μ)          
            if opt_err_μ <= options.κ_ϵ * data.μ
                data.μ = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ, data.μ ^ options.θ_μ))
                # println("reset filter")
                reset_filter!(data)
                # performance of current iterate updated to account for barrier parameter change
                data.barrier_obj_curr = barrier_objective!(problem, data, mode=:nominal)
                constraint!(problem, data.μ; mode=:nominal)
                data.primal_1_curr = constraint_violation_1norm(problem, mode=:nominal)
                data.j += 1
                continue
            end
            
            options.verbose && iteration_status(data, options)
            data.p = 0
            
            forward_pass!(policy, problem, data, options, verbose=options.verbose)
            data.status != 0 && break
            
            # # for watchdog/filter reset heuristic (pg 13 IPOPT)
            # if data.l >= 2
            #     data.t += 1
            # else
            #     data.t = 0
            # end
            rescale_duals!(problem, data.μ, options)
            update_nominal_trajectory!(problem)
            (!data.armijo_passed && !data.switching) && update_filter!(data, options)
            data.barrier_obj_curr = data.barrier_obj_next
            data.primal_1_curr = data.primal_1_next
        end
        
        data.k += 1
        data.wall_time += iter_time
    end
    display(problem.nominal_states)
    
    options.verbose && iteration_status(data, options)
    if data.k == options.max_iterations 
        data.status = 8
        options.verbose && @warn "Maximum solver iterations reached."
    end
    return nothing
end

function update_filter!(data::SolverData, options::Options)
    new_filter_pt = [(1. - options.γ_θ) * data.primal_1_curr,
                        data.barrier_obj_curr - options.γ_φ * data.primal_1_curr]
    push!(data.filter, new_filter_pt)
end

function reset_filter!(data::SolverData)
    empty!(data.filter)
    push!(data.filter, [data.max_primal_1, -Inf])
    data.status = 0
end

function optimality_error(policy::PolicyData, problem::ProblemData, solver::SolverData, options::Options, μ::Float64; mode=:nominal)
    dual_inf::Float64 = 0     # dual infeasibility (stationarity of Lagrangian)
    primal_inf::Float64 = 0   # constraint violation (primal infeasibility)
    cs_inf::Float64 = 0       # complementary slackness violation
    ϕ_norm::Float64 = 0       # norm of dual equality
    v_norm::Float64 = 0       # norm of dual inequality
    
    N = problem.horizon
    constr_data = problem.constr_data
    _, _, h, il, iu = primal_trajectories(problem, mode=mode)
    ϕ, vl, vu = dual_trajectories(problem, mode=mode)
    
    Qu = policy.action_value.gradient_action
    hu = constr_data.jacobian_action

    num_ineq = 0
    
    for k = N-1:-1:1
        constr = constr_data.constraints[k]

        # dual infeasibility (stationarity)
        policy.u_tmp[k] .= Qu[k]
        mul!(policy.u_tmp[k], transpose(hu[k]), ϕ[k], 1.0, 1.0)
        dual_inf = max(dual_inf, norm(policy.u_tmp[k], Inf))
        ϕ_norm += norm(ϕ[k], 1)

        # primal feasibility (eq. constraint satisfaction)
        primal_inf = max(primal_inf, norm(h[k], Inf))

        # complementary slackness
        for i = 1:constr.num_action
            if !isinf(il[k][i])
                cs_inf = max(cs_inf, il[k][i] * vl[k][i])
                v_norm += vl[k][i]
                num_ineq += 1
            end
            if !isinf(iu[k][i])
                cs_inf = max(cs_inf, iu[k][i] * vu[k][i])
                v_norm += vu[k][i]
                num_ineq += 1
            end
        end
    end
    cs_inf -= μ
    
    scaling_cs = max(options.s_max, v_norm / max(num_ineq, 1.0))  / options.s_max
    scaling_dual = max(options.s_max, (ϕ_norm + v_norm) / max(num_ineq + constr_data.num_constraints[1], 1.0))  / options.s_max
    return dual_inf / scaling_dual, primal_inf, cs_inf / scaling_cs
end

function rescale_duals!(problem::ProblemData, μ::Float64, options::Options)
    N = problem.horizon
    constr_data = problem.constr_data
    κ_Σ = options.κ_Σ
    il = problem.ineq_lower
    iu = problem.ineq_upper
    vl = problem.ineq_duals_lo
    vu = problem.ineq_duals_up
    for k = 1:N-1
        for i = constr_data.constraints[k].num_action
            if !isinf(il[k][i])
                vl[k][i] = max(min(vl[k][i], κ_Σ * μ / il[k][i]), μ / (κ_Σ *  il[k][i]))
            end
            if !isinf(iu[k][i])
                vu[k][i] = max(min(vu[k][i], κ_Σ * μ / iu[k][i]), μ / (κ_Σ *  iu[k][i]))
            end
        end
    end
end

function reset_duals!(problem::ProblemData)
    N = problem.horizon
    constr_data = problem.constr_data
    bounds = problem.bounds
    for k = 1:N-1
        fill!(problem.eq_duals[k], 0.0)
        fill!(problem.nominal_eq_duals[k], 0.0)
        constr = constr_data.constraints[k]
        for i = 1:constr.num_action
            problem.ineq_duals_lo[k][i] = isinf(bounds[k].lower[i]) ? 0.0 : 1.0
            problem.ineq_duals_up[k][i] = isinf(bounds[k].upper[i]) ? 0.0 : 1.0
            problem.nominal_ineq_duals_lo[k][i] = isinf(bounds[k].lower[i]) ? 0.0 : 1.0
            problem.nominal_ineq_duals_up[k][i] = isinf(bounds[k].upper[i]) ? 0.0 : 1.0
        end
    end
end

