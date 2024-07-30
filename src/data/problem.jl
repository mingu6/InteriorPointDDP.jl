mutable struct ProblemData#{T,X,U,H,D,O,FX,FU,FW,OX,OU,OXX,OUU,OUX,C,CX,CU}
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

    # model data
    model::ModelData#{T,FX,FU}

    # objective/cost data
    cost_data::CostsData#{O,OX,OU,OXX,OUU,OUX}
    
    # constraints data
    constr_data::ConstraintsData#{T,C,CX,CU}

    # trajectory: z = (x1,..., xT, u1,..., uT-1) | Δz = (Δx1..., ΔxT, Δu1,..., ΔuT-1)
    trajectory::Vector#{T}

    horizon::Int
end

function problem_data(dynamics::Model{T}, costs::Costs{T}, constraints::Constraints{T}) where T

    @assert length(dynamics) + 1 == length(costs)

	states = [[zeros(d.num_state) for d in dynamics]..., 
            zeros(dynamics[end].num_next_state)]
    actions = [[zeros(d.num_action) for d in dynamics]..., zeros(0)]
    constr = [zeros(h.num_constraint) for h in constraints]
    ineq_lower = [zeros(d.num_action) for d in dynamics]
    ineq_upper = [zeros(d.num_action) for d in dynamics]
    eq_duals = [zeros(h.num_constraint) for h in constraints]
    ineq_duals_lo = [ones(d.num_action) for d in dynamics]
    ineq_duals_up = [ones(d.num_action) for d in dynamics]

    nominal_states = [[zeros(d.num_state) for d in dynamics]..., 
                zeros(dynamics[end].num_next_state)]
    nominal_actions = [[zeros(d.num_action) for d in dynamics]..., zeros(0)]
    nominal_constr = [zeros(h.num_constraint) for h in constraints]
    nominal_ineq_lower = [zeros(d.num_action) for d in dynamics]
    nominal_ineq_upper = [zeros(d.num_action) for d in dynamics]
    nominal_eq_duals = [zeros(h.num_constraint) for h in constraints]
    nominal_ineq_duals_lo = [ones(d.num_action) for d in dynamics]
    nominal_ineq_duals_up = [ones(d.num_action) for d in dynamics]

    model = model_data(dynamics)
    costs_dat = costs_data(dynamics, costs)
    constr_data = constraint_data(constraints)

    trajectory = zeros(num_trajectory(dynamics))
    
    horizon = length(costs)
    
    ProblemData(states, actions, constr, ineq_lower, ineq_upper, eq_duals, ineq_duals_lo, ineq_duals_up,
        nominal_states, nominal_actions, nominal_constr, nominal_ineq_lower, nominal_ineq_upper, nominal_eq_duals,
        nominal_ineq_duals_lo, nominal_ineq_duals_up, model, costs_dat, constr_data,
        trajectory, horizon)
end
