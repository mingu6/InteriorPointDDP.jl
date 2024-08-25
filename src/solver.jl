"""
    Problem Data
"""

mutable struct Solver{T}#,N,M,NN,MM,MN,NNN,MNN,X,U,H,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
    problem::ProblemData#{T,X,U,H,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
	policy::PolicyData#{N,M,NN,MM,MN,NNN,MNN}
	data::SolverData{T}
    options::Options{T}
end

function Solver(dynamics::Vector{Dynamics{T}}, costs::Costs{T}, constraints::Constraints{T, J}=nothing,
                bounds::Bounds{T}=nothing; options=Options{T}()) where {T, J}

    N = length(costs)
    if isnothing(constraints)
        constraint = Constraint()
        constraints = [constraint for k = 1:N-1]
    end
    if isnothing(bounds)
        bounds = [Bound(T, dynamics[k].num_action) for k = 1:N-1]
    end

    # allocate policy data
    policy = policy_data(dynamics, constraints)

    # allocate model data
    problem = problem_data(dynamics, costs, constraints, bounds)

    # allocate solver data
    data = solver_data()
    
	Solver(problem, policy, data, options)
end

function get_trajectory(solver::Solver)
	return solver.problem.nominal_states, solver.problem.nominal_actions[1:end-1]
end

function current_trajectory(solver::Solver)
	return solver.problem.states, solver.problem.actions[1:end-1]
end

function initialize_trajectory!(solver::Solver, actions, x1)
    bounds = solver.problem.bounds
    constraints = solver.problem.constr_data.constraints
    dynamics = solver.problem.model.dynamics
    options = solver.options
    solver.problem.nominal_states[1] .= x1
    for (t, ut) in enumerate(actions)
        lb = bounds[t].lower
        ub = bounds[t].upper
        nh = length(lb)
        for i = 1:nh
            if isinf(lb[i]) && isinf(ub[i])
                solver.problem.nominal_actions[t][i] = ut[i]
            elseif isinf(lb[i]) && !isinf(ub[i])
                solver.problem.nominal_actions[t][i] = min(ut[i], ub[i] - options.κ_1 * max(1, abs(ub[i])))
            elseif !isinf(lb[i]) && isinf(ub[i])
                solver.problem.nominal_actions[t][i] = max(ut[i], lb[i] + options.κ_1 * max(1, abs(lb[i])))
            else
                p_L = min(options.κ_1 * max(1, abs(lb[i])), options.κ_2 * (ub[i] - lb[i]))
                p_U = min(options.κ_1 * max(1, abs(ub[i])), options.κ_2 * (ub[i] - lb[i]))
                if ut[i] < lb[i] + p_L
                    solver.problem.nominal_actions[t][i] = lb[i] + p_L
                elseif ut[i] > ub[i] - p_U
                    solver.problem.nominal_actions[t][i] = ub[i] - p_U
                else
                    solver.problem.nominal_actions[t][i] = ut[i]
                end
            end
        end
        solver.problem.nominal_states[t+1] .= dynamics!(dynamics[t], 
                                solver.problem.nominal_states[t], solver.problem.nominal_actions[t])
    end
    # display(actions)
end

# TODO: initialize_duals
