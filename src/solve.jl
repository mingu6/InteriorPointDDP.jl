"""
    gradient of Lagrangian
        https://web.stanford.edu/class/ee363/lectures/lqr-lagrange.pdf
"""
function lagrangian_gradient!(data::SolverData, policy::PolicyData, problem::ProblemData)
	p = policy.value.gradient
    Qx = policy.action_value.gradient_state
    Qu = policy.action_value.gradient_action
    H = length(problem.states)

    for t = 1:H-1
        Lx = @views data.gradient[data.indices_state[t]]
        Lx .= Qx[t]
        Lx .-= p[t]
        Lu = @views data.gradient[data.indices_action[t]]
        Lu .= Qu[t]
        # data.gradient[data.indices_state[t]] = Qx[t] - p[t] # should always be zero by construction
        # data.gradient[data.indices_action[t]] = Qu[t]
    end
    # NOTE: gradient wrt x1 is satisfied implicitly
end

# function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Objective{T}}
#     ilqr_solve!(solver, args...; kwargs...)
# end

function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:InteriorPoint{T}}
    ipddp_solve!(solver, args...; kwargs...)
end

function ipddp_solve!(solver::Solver, states, actions; kwargs...)
    initialize_controls!(solver, actions)
    initialize_states!(solver, states)
    ipddp_solve!(solver; kwargs...)
end


function ipddp_solve!(solver::Solver; 
    iteration=true)

    # START TIMER
    costs = []
    steps = []
    (solver.options.verbose && iteration==1) && solver_info()

	# data
	policy = solver.policy
    problem = solver.problem
    reset!(problem.model)
    reset!(problem.objective)
	data = solver.data
    solver.options.reset_cache && reset!(data)
    options = solver.options
    constraints = solver.problem.objective.costs.constraint_data

    # initial rollout is performed in caller file
	initial_cost = cost!(data, problem,
        mode=:nominal)
    constraint!(constraints, problem.states, problem.actions, problem.parameters)
    if data.perturbation == 0
        initial_cost = initial_cost[1] # = data.objective[1] which is the obj func/cost for first iteration
        n_minus_1 = problem.horizon - 1
        num_inequals = constraints.constraints[1].num_inequality
        data.perturbation = initial_cost/n_minus_1/num_inequals
    end

    reset_filter!(problem, data, options)
    reset_regularisation!(data, options)

    time = 0
    for iter = 1:solver.options.max_iterations
        iter_time = @elapsed begin   
            gradients!(problem,
        mode=:nominal)
        backward_pass!(policy, problem, data, options)
        forward_pass!(policy, problem, data, options,
            min_step_size=solver.options.min_step_size,
            line_search=solver.options.line_search,
            verbose=solver.options.verbose)
        end
    
        # info
        data.iterations[1] += 1
        if iter % 10 == 1
            println("")
            println(rpad("Iteration", 15), rpad("Elapsed time", 15), rpad("mu", 15), rpad("Cost", 15), rpad("Opt.error", 15), rpad("Reg.power", 13), rpad("Stepsize", 15))
        end
        if solver.options.verbose
            println(
                rpad(string(iter), 15), 
                rpad(@sprintf("%.5e", time+=iter_time), 15), 
                rpad(@sprintf("%.5e", data.perturbation), 15), 
                rpad(@sprintf("%.5e", data.objective[1]), 15), 
                rpad(@sprintf("%.5e", options.opterr), 15), 
                rpad(@sprintf("%.3e", options.reg), 13), 
                rpad(@sprintf("%.5e", data.step_size[1]), 15)
            )            
        end 

        push!(costs, data.objective[1])
        push!(steps, data.step_size[1])

        # check convergence
        if max(options.opterr, data.perturbation) <= options.objective_tolerance
            println("~~~~~~~~~~~~~~~~~~~")
            println("Optimality reached!")
            return nothing
        end

        if options.opterr <= 0.2 * data.perturbation
            data.perturbation = max(options.objective_tolerance/10.0, min(0.2 * data.perturbation, data.perturbation^1.2))
        end
    end

    return nothing
end


function reset_filter!(problem::ProblemData, data::SolverData, options::Options) 
    constraint_vals = problem.objective.costs.constraint_data.violations
    cost = data.objective[1]
    perturbation = data.perturbation
    if !options.feasible
        slacks = problem.objective.costs.constraint_data.slacks
        data.logcost = cost - perturbation * sum(log.(vcat(slacks...)))
        # Add padding
        flattened_sum = vcat([(length(s) > length(c) ? vcat(c, zeros(length(s) - length(c))) : c) + 
                      (length(s) < length(c) ? vcat(s, zeros(length(c) - length(s))) : s) 
                      for (s, c) in zip(slacks, constraint_vals)]...)
        data.err = norm(flattened_sum, 1)
        if data.err < options.objective_tolerance
            data.err = 0
        end
    else
        data.logcost = cost - perturbation * sum(log.(vcat((-1 .* constraint_vals)...)))
        data.err = 0
    end
    data.filter = [data.logcost, data.err]
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