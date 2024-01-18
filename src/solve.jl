function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Costs{T}}
    ipddp_solve!(solver, args...; kwargs...)
end

function ipddp_solve!(solver::Solver, states, actions; kwargs...)
    initialize_controls!(solver, actions)
    initialize_states!(solver, states)
    ipddp_solve!(solver; kwargs...)
end

function ipddp_solve!(solver::Solver; iteration=true)
    costs = []
    steps = []
    (solver.options.verbose && iteration==1) && solver_info()

	# data
	policy = solver.policy
    problem = solver.problem
    reset!(problem.model)
    reset!(problem.costs)
	data = solver.data
    options = solver.options
    options.reset_cache && reset!(data)
    constr_data = solver.problem.constraints

    H = problem.horizon
    constraint!(constr_data, problem.nominal_states, problem.nominal_actions, problem.parameters)
    # update constraint evaluations for nominal trajectory
    if options.feasible
        for t = 1:H-1
            constr_data.nominal_inequalities[t] .= constr_data.inequalities[t]
        end
    end
    
    # initialize θ_max to initialise filter parameters for infeasible IPDDP
    θ_0 = options.feasible ? 0.0 : constraint_violation_1norm(constr_data)
    data.θ_max = !options.feasible ? 1e4 * max(1.0, θ_0) : 0.0
    data.θ_min = !options.feasible ? 1e-4 * max(1.0, θ_0) : 0.0
    data.constr_viol_norm = θ_0
    
    cost!(data, problem, mode=:nominal)[1]
    data.μ_j = data.μ_j == 0.0 ? options.μ_0 * data.costs[1] / H : data.μ_j
    data.barrier_obj = barrier_objective!(problem, data, options.feasible, mode=:nominal)

    reset_filter!(data, options)
    reset_regularisation!(data, options)

    time = 0
    while data.k <= options.max_iterations
        iter_time = @elapsed begin
            gradients!(problem, mode=:nominal)
            backward_pass!(policy, problem, data, options)
            
            # check (outer) overall problem convergence
            opt_err = optimality_error(policy, problem, options, data.μ_j)
            data.optimality_error = opt_err
            max(opt_err, data.μ_j) <= options.optimality_tolerance && break
            
            # check (inner) barrier problem convergence
            if opt_err <= options.κ_ϵ * data.μ_j
                data.μ_j = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ_j, data.μ_j ^ options.θ_μ))
                reset_filter!(data, options)
                data.barrier_obj = barrier_objective!(problem, data, options.feasible, mode=:nominal)
                data.j += 1
                if data.k == 1
                    continue
                end
            end
            
            # forward_pass!(policy, problem, data, options, min_step_size=options.min_step_size, verbose=options.verbose)
            forward_pass!(policy, problem, data, options, verbose=options.verbose)
            !data.status[1] && break  # exit if line search failed
        end
        # info
        data.iterations[1] += 1
        if data.k % 10 == 1
            println("")
            println(rpad("Iteration", 15), rpad("Elapsed time", 15), rpad("μ", 15), rpad("Cost", 15), rpad("Opt.error", 15), rpad("Reg.power", 13), rpad("Stepsize", 15))
        end
        if options.verbose
            println(
                rpad(string(data.k), 15), 
                rpad(@sprintf("%.5e", time+=iter_time), 15), 
                rpad(@sprintf("%.5e", data.μ_j), 15), 
                rpad(@sprintf("%.5e", data.costs[1]), 15), 
                rpad(@sprintf("%.5e", data.optimality_error), 15), 
                rpad(@sprintf("%.3e", options.reg), 13), 
                rpad(@sprintf("%.5e", data.step_size[1]), 15)
            )            
        end 

        push!(costs, data.costs[1])
        push!(steps, data.step_size[1])
        data.k += 1
    end
    return nothing
end

function reset_filter!(data::SolverData, options::Options) 
    if !options.feasible
        data.filter = [[data.θ_max, -Inf]]
    else
        data.filter = [[0.0, Inf]]
    end
    data.status[1] = true
end

function reset_regularisation!(data::SolverData, options::Options)
    # TODO: Is this needed or not?
    options.start_reg = 0
    options.end_reg = 24 
    options.reg_step = 1
    data.status[1] = true
    options.recovery = 0.0
end

function optimality_error(policy::PolicyData, problem::ProblemData, options::Options, μ_j::Float64)
    stat_err::Float64 = 0   # stationarity of Lagrangian
    viol_err::Float64 = 0   # constraint violation (equality and slacks + ineq)
    cs_err::Float64 = 0     # complementary slackness
    s_norm::Float64 = 0     # optimality error rescaling term
    
    H = length(problem.states)
    Qu = policy.action_value.gradient_action
    constr_data = problem.constraints
    c = constr_data.inequalities
    s = constr_data.ineq_duals
    y = constr_data.slacks
    cy = options.feasible ? c : y
    !options.feasible && (μ_j *= -1)
    
    for t = 1:H-1
        stat_err = max(stat_err, norm(Qu[t], Inf))
        num_inequality = constr_data.constraints[t].num_inequality
        for i = 1:num_inequality
            cs_err = max(cs_err, abs(s[t][i] * cy[t][i] + μ_j))
        end
        if !options.feasible
            for i = 1:num_inequality
                viol_err = max(viol_err, abs(c[t][i] + y[t][i]))
            end
        end
        s_norm += norm(s[t], 1)
    end
    
    s_d = max(options.s_max, s_norm / (H * length(s[1])))  / options.s_max
    optimality_error = options.feasible ? max(stat_err / s_d, cs_err / s_d) : max(stat_err / s_d, viol_err, cs_err / s_d)
    return optimality_error
end
