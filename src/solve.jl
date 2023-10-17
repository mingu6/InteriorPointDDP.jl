function ilqr_solve!(solver::Solver; 
    iteration=true)

    (solver.options.verbose && iteration==1) && solver_info()

	# data
	policy = solver.policy
    problem = solver.problem
    reset!(problem.model)
    reset!(problem.objective)
	data = solver.data
    solver.options.reset_cache && reset!(data)

	cost!(data, problem,
        mode=:nominal)
    gradients!(problem,
        mode=:nominal)
    backward_pass!(policy, 
        problem,
        solver.data,
        solver.options)

    obj_prev = data.objective[1]
    for i = 1:solver.options.max_iterations
        forward_pass!(policy, problem, data, constraints,
            min_step_size=solver.options.min_step_size,
            line_search=solver.options.line_search,
            verbose=solver.options.verbose)
        if solver.options.line_search != :none
            gradients!(problem,
                mode=:nominal)
            backward_pass!(policy, problem,
                mode=:nominal)
            lagrangian_gradient!(data, policy, problem) # get rid of this, not required
        end

        # gradient norm
        gradient_norm = norm(data.gradient, Inf)

        # info
        data.iterations[1] += 1
        solver.options.verbose && println(
            "iter:                  $i
             cost:                  $(data.objective[1])
			 gradient_norm:         $(gradient_norm)
			 max_violation:         $(data.max_violation[1])
			 step_size:             $(data.step_size[1])")

        # check convergence
		gradient_norm < solver.options.lagrangian_gradient_tolerance && break
        abs(data.objective[1] - obj_prev) < solver.options.objective_tolerance ? break : (obj_prev = data.objective[1])
        !data.status[1] && break
    end

    return nothing
end

function ilqr_solve!(solver::Solver, states, actions; kwargs...)
    initialize_controls!(solver, actions)
    initialize_states!(solver, states)
    ilqr_solve!(solver; kwargs...)
end


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

"""
    augmented Lagrangian solve
"""
function constrained_ilqr_solve!(solver::Solver; augmented_lagrangian_callback!::Function=x->nothing)

	solver.options.verbose && solver_info()

    # reset solver cache
    reset!(solver.data)

    # reset duals
    for (t, λ) in enumerate(solver.problem.objective.costs.constraint_dual)
        fill!(λ, 0.0)
	end

	# initialize penalty
	for (t, ρ) in enumerate(solver.problem.objective.costs.constraint_penalty)
        fill!(ρ, solver.options.initial_constraint_penalty)
	end

	for i = 1:solver.options.max_dual_updates
		solver.options.verbose && println("  al iter: $i")

		# primal minimization
		ilqr_solve!(solver, 
            iteration=i)

		# update trajectories
		cost!(solver.data, solver.problem,
            mode=:nominal)

        # constraint violation
		solver.data.max_violation[1] <= solver.options.constraint_tolerance && break

        # dual ascent
        ## here is where you update mu
		augmented_lagrangian_update!(solver.problem.objective.costs,
			scaling_penalty=solver.options.scaling_penalty,
            max_penalty=solver.options.max_penalty)

		# user-defined callback (continuation methods on the models etc.)
		augmented_lagrangian_callback!(solver)
	end

    return nothing
end

function constrained_ilqr_solve!(solver::Solver, states, actions; kwargs...)
    initialize_controls!(solver, actions)
    initialize_states!(solver, states)
    constrained_ilqr_solve!(solver; kwargs...)
end

function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:Objective{T}}
    ilqr_solve!(solver, args...; kwargs...)
end

function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:AugmentedLagrangianCosts{T}}
    constrained_ilqr_solve!(solver, args...; kwargs...)
end

function solve!(solver::Solver{T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O}, args...; kwargs...) where {T,N,M,NN,MM,MN,NNN,MNN,X,U,D,O<:InteriorPoint{T}}
    ipddp_solve!(solver, args...; kwargs...)
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

    if data.perturbation == 0
        initial_cost = initial_cost[1] # = data.objective[1] which is the obj func/cost for first iteration
        horizon = problem.horizon
        num_inequals = constraints.constraints[1].num_inequality
        data.perturbation = initial_cost/horizon/num_inequals
    end

    reset_filter!(problem, data, options)
    reset_regularisation!(data, options)

    for i = 1:solver.options.max_iterations
        iter_time = @elapsed begin   
            gradients!(problem,
        mode=:nominal)
        backward_pass!(policy, problem, options)
        forward_pass!(policy, problem, data, options,
            min_step_size=solver.options.min_step_size,
            line_search=solver.options.line_search,
            verbose=solver.options.verbose)
        end
    
        # info
        pad_width = 12
        data.iterations[1] += 1
        if iter % 10 == 1
            println(rpad("Iteration", 5), rpad("Time", 10), rpad("mu", 10), rpad("Cost", 10), rpad("Opt.error", 10), rpad("Reg.power", 5), rpad("Stepsize", 12))
        end
        if solver.options.verbose
            println(rpad(string(i), 5), rpad(string(iter_time), 10), rpad(string(data.perturbation), 10), rpad(string(data.objective[1]), 10), rpad(string(options.opterr), 10), rpad(string(options.reg), 5), rpad(string(data.step_size[1]), 12))
        end 

        push!(costs, data.objective[1])
        push!(steps, data.step_size[1])

        # check convergence
        if max(options.opterr, data.perturbation) <= options.objective_tolerance
            print("Optimality reached")
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
        # TODO: Less than or not less than?
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
    options.reg = 0
    data.status[1] = true
    options.recovery = 0.0
end