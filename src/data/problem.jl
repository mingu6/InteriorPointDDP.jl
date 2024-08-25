struct Bound{T}
    lower::Vector{T}
    upper::Vector{T}
end

function Bound(T, num_control::Int)
    return Bound(-Inf .* ones(T, num_control), Inf .* ones(T, num_control))
end

function Bound(num_control::Int, lower::T, upper::T) where T
    return Bound(lower .* ones(T, num_control), upper .* ones(T, num_control))
end

Bounds{T} = Vector{Bound{T}}

mutable struct ProblemData{T, J}
    # current trajectory
    states#::Vector{X}
    actions#::Vector{U}
    constraints#::Vector{H} 
    ineq_lower#::Vector{U}
    ineq_upper#::Vector{U}
    eq_duals#::Vector{H}
    ineq_duals_lo#::Vector{U}
    ineq_duals_up#::Vector{U}

    # nominal trajectory
    nominal_states#::Vector{X}
    nominal_actions#::Vector{U}
    nominal_constraints#::Vector{H}
    nominal_ineq_lower#::Vector{U}
    nominal_ineq_upper#::Vector{U}
    nominal_eq_duals#::Vector{H}
    nominal_ineq_duals_lo#::Vector{U}
    nominal_ineq_duals_up#::Vector{U}

    model_fn::Model{T, J}
    constraints_fn::Constraints{T, J}

    # model data
    model_data::Vector{DynConExpansion{T}}

    # objective/cost data
    cost_data::CostsData#{O,OX,OU,OXX,OUU,OUX}
    
    # constraints data
    constr_data::Vector{DynConExpansion{T}}

    # bounds data
    bounds::Bounds{T}

    # trajectory: z = (x1,..., xT, u1,..., uT-1) | Δz = (Δx1..., ΔxT, Δu1,..., ΔuT-1)
    trajectory::Vector#{T}

    horizon::Int
end

function problem_data(dynamics::Model{T, J}, costs::Costs{T}, constraints::Constraints{T, J}, bounds::Bounds{T}) where {T, J}

    @assert length(dynamics) + 1 == length(costs)

	states = [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]
    actions = [[zeros(d.num_control) for d in dynamics]..., zeros(0)]
    constr = [zeros(h.num_constraint) for h in constraints]
    ineq_lower = [zeros(d.num_control) for d in dynamics]
    ineq_upper = [zeros(d.num_control) for d in dynamics]
    eq_duals = [zeros(h.num_constraint) for h in constraints]
    ineq_duals_lo = [ones(d.num_control) for d in dynamics]
    ineq_duals_up = [ones(d.num_control) for d in dynamics]

    nominal_states = [[zeros(d.num_state) for d in dynamics]..., 
                zeros(dynamics[end].num_next_state)]
    nominal_actions = [[zeros(d.num_control) for d in dynamics]..., zeros(0)]
    nominal_constr = [zeros(h.num_constraint) for h in constraints]
    nominal_ineq_lower = [zeros(d.num_control) for d in dynamics]
    nominal_ineq_upper = [zeros(d.num_control) for d in dynamics]
    nominal_eq_duals = [zeros(h.num_constraint) for h in constraints]
    nominal_ineq_duals_lo = [ones(d.num_control) for d in dynamics]
    nominal_ineq_duals_up = [ones(d.num_control) for d in dynamics]

    model_dat = map(d -> DynConExpansion(d), dynamics)
    costs_dat = costs_data(dynamics, costs)
    constr_data = map(c -> DynConExpansion(c), constraints)

    trajectory = zeros(num_trajectory(dynamics))
    
    horizon = length(costs)
    
    ProblemData(states, actions, constr, ineq_lower, ineq_upper, eq_duals, ineq_duals_lo, ineq_duals_up,
        nominal_states, nominal_actions, nominal_constr, nominal_ineq_lower, nominal_ineq_upper, nominal_eq_duals,
        nominal_ineq_duals_lo, nominal_ineq_duals_up, dynamics, constraints, model_dat, costs_dat, constr_data, bounds,
        trajectory, horizon)
end

function evaluate_derivatives!(problem::ProblemData, quasi_newton::Bool; mode=:nominal)
    model_exp = problem.model_data
    cons_exp = problem.constr_data
    model = problem.model_fn
    cons = problem.constraints_fn
    x, u, _, _, _ = primal_trajectories(problem, mode=mode)
    ϕ, _, _ = dual_trajectories(problem, mode=:mode)
    gradients!(problem.cost_data.costs, problem, mode=mode)
    for (k, d) in enumerate(model)
        jacobian!(model_exp[k], d, x[k], u[k])
        jacobian!(cons_exp[k], cons[k], x[k], u[k])
        !quasi_newton && second_order_contraction!(cons_exp[k], cons[k], x[k], u[k], ϕ[k])
    end
end
