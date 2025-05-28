"""
    Problem Data
"""
mutable struct Solver{T}
    problem::ProblemData{T}
    update_rule::UpdateRuleData{T}
	data::SolverData{T}
    options::Options{T}
end

function Solver(T, dynamics::Vector{Dynamics}, objectives::Objectives, constraints::Constraints,
            bounds::Union{Bounds, Nothing}=nothing; options::Union{Options, Nothing}=nothing)

    # allocate update rule data  
    update_rule = update_rule_data(T, constraints)

    # allocate model data
    problem = problem_data(T, dynamics, objectives, constraints, bounds)

    # allocate solver data
    data = solver_data(T)

    options = isnothing(options) ? Options{T}() : options

	Solver(problem, update_rule, data, options)
end

function Solver(T, dynamics::Vector{Dynamics}, objectives::Objectives,
            bounds::Union{Bounds, Nothing}=nothing; options::Union{Options, Nothing}=nothing)
    constraints = [Constraint(o.num_state, o.num_control) for o in objectives]
    
    # allocate update rule data  
    update_rule = update_rule_data(T, dynamics, constraints)

    # allocate model data
    problem = problem_data(T, dynamics, objectives, constraints, bounds)

    # allocate solver data
    data = solver_data(T)

    options = isnothing(options) ? Options{T}() : options

	Solver(T, problem, update_rule, data, options)
end

function get_trajectory(solver::Solver{T}) where T
	return solver.problem.nominal_states, solver.problem.nominal_controls
end

function current_trajectory(solver::Solver{T}) where T
	return solver.problem.states, solver.problem.controls
end

function initialize_trajectory!(solver::Solver{T}, controls::Vector{Vector{T}}, x1::Vector{T}) where T
    bounds = solver.problem.bounds
    dynamics = solver.problem.model.dynamics
    options = solver.options
    N = solver.problem.horizon

    solver.problem.nominal_states[1] .= x1
    ū = solver.problem.nominal_controls
    x̄ = solver.problem.nominal_states
    ūl = solver.problem.nominal_ineq_lo
    ūu = solver.problem.nominal_ineq_up

    u_tmp1 = solver.update_rule.u_tmp1
    u_tmp2 = solver.update_rule.u_tmp2

    for t = 1:N
        # project primal variables within Bounds

        nu = length(controls[t])
        for i in 1:nu
            if !isinf(bounds[t].lower[i]) && isinf(bounds[t].upper[i])
                u_tmp1[t][i] = max(bounds[t].lower[i], 1.0)
                u_tmp1[t][i] *= options.κ_1
                u_tmp1[t][i] += bounds[t].lower[i]
                ū[t][i] = max(controls[t][i], u_tmp1[t][i])
            elseif !isinf(bounds[t].upper[i]) && isinf(bounds[t].lower[i])
                u_tmp1[t][i] = max(bounds[t].upper[i], 1.0)
                u_tmp1[t][i] *= -options.κ_1
                u_tmp1[t][i] += bounds[t].upper[i]
                replace!(u_tmp1[t][i], NaN=>Inf)
                ū[t][i] = min(controls[t][i], u_tmp1[t])
            elseif !isinf(bounds[t].upper[i]) && !isinf(bounds[t].lower[i])
                u_tmp1[t][i] = bounds[t].lower[i] + min(options.κ_1 * max(1.0, abs(bounds[t].lower[i])),
                                    options.κ_2 * (bounds[t].upper[i] - bounds[t].lower[i]))
                u_tmp2[t][i] = bounds[t].upper[i] - min(options.κ_1 * max(1.0, abs(bounds[t].upper[i])),
                                    options.κ_2 * (bounds[t].upper[i] - bounds[t].lower[i]))
                ū[t][i] = min(max(controls[t][i], u_tmp1[t][i]), u_tmp2[t][i])
            else
                ū[t][i] = controls[t][i]
            end
        end

        # initialise inequality constraints

        ūl[t] .= ū[t]
        ūl[t] .-= bounds[t].lower
        ūu[t] .= bounds[t].upper
        ūu[t] .-= ū[t]

        t < N && dynamics!(dynamics[t], x̄[t+1], x̄[t], ū[t])
    end
end

