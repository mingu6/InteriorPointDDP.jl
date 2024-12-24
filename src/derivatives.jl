function evaluate_derivatives!(dynamics::Model, data::ProblemData{T}; mode=:nominal) where T
    x, u, _, _, _ = primal_trajectories(data, mode=mode)
    fx = data.model.jacobian_state
    fu = data.model.jacobian_control
    jacobian!(fx, fu, dynamics, x, u)
end

function evaluate_derivatives!(costs::Costs, data::ProblemData{T}; mode=:nominal) where T
    x, u, _, _, _ = primal_trajectories(data, mode=mode)
    lx = data.cost_data.gradient_state
    lu = data.cost_data.gradient_control
    lxx = data.cost_data.hessian_state_state
    luu = data.cost_data.hessian_control_control
    lux = data.cost_data.hessian_control_state
    cost_gradient!(lx, lu, costs, x, u)
    cost_hessian!(lxx, luu, lux, costs, x, u) 
end

function evaluate_derivatives!(constraints_data::ConstraintsData{T}, problem::ProblemData{T}; mode=:nominal) where T
    x, u, _, _, _ = primal_trajectories(problem, mode=mode)
    ϕ = mode == :nominal ? problem.nominal_eq_duals : problem.eq_duals
    hx = constraints_data.jacobian_state
    hu = constraints_data.jacobian_control
    vhxx = constraints_data.vhxx
    vhux = constraints_data.vhux
    vhuu = constraints_data.vhuu
    jacobian!(hx, hu, constraints_data.constraints, x, u)
    tensor_contraction!(vhxx, vhux, vhuu, constraints_data.constraints, x, u, ϕ)
end

function evaluate_derivatives!(problem::ProblemData{T}; mode=:nominal) where T
    evaluate_derivatives!(problem.model.dynamics, problem, mode=mode)
    evaluate_derivatives!(problem.cost_data.costs, problem, mode=mode)
    evaluate_derivatives!(problem.constraints_data, problem, mode=mode)
end
