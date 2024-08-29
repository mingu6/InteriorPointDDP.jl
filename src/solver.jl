"""
    Problem Data
"""
mutable struct Solver{T}#,N,M,NN,MM,MN,NNN,MNN,X,U,H,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
    problem::ProblemData#{T,X,U,H,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
	policy::PolicyData#{N,M,NN,MM,MN,NNN,MNN}
	data::SolverData{T}
    options::Options{T}
end

function Solver(dynamics::Vector{Dynamics{T}}, costs::Costs{T}, constraints::Constraints{T}, bounds=nothing; options=Options{T}()) where T

    # allocate policy data  
    policy = policy_data(dynamics, constraints)

    # allocate model data
    problem = problem_data(dynamics, costs, constraints, bounds)

    # allocate solver data
    data = solver_data()

	Solver(problem, policy, data, options)
end

function Solver(dynamics::Vector{Dynamics{T}}, costs::Costs{T}) where T
    N = length(costs)
    constraint = Constraint()
    constraints = [constraint for k = 1:N-1]
    
    # allocate policy data  
    policy = policy_data(dynamics, constraints)

    # allocate model data
    problem = problem_data(dynamics, costs, constraints)

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
    dynamics = solver.problem.model.dynamics
    options = solver.options
    solver.problem.nominal_states[1] .= x1
    ū = solver.problem.nominal_actions
    x̄ = solver.problem.nominal_states

    for (k, uk) in enumerate(actions)
        # project primal variables within bounds

        bk = bounds[k]
        bkl = bounds[k].lower[bk.indices_lower]
        bku = bounds[k].upper[bk.indices_upper]

        ū[k] .= uk
        ū[k][bk.indices_lower] .= max.(uk[bk.indices_lower], bkl + options.κ_1 .* max.(1., bkl))
        ū[k][bk.indices_upper] .= min.(uk[bk.indices_upper], bku - options.κ_1 .* max.(1., bku))

        dynamics!(dynamics[k], x̄[k+1], x̄[k], ū[k])
    end
end

