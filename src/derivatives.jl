function evaluate_derivatives!(dynamics::Model, data::ProblemData{T}; mode=:nominal) where T
    x, u, _, _, _ = primal_trajectories(data, mode=mode)
    fx = data.model.jacobian_state
    fu = data.model.jacobian_control
    jacobian!(fx, fu, dynamics, x, u)
end

function evaluate_derivatives!(objectives::Objectives, data::ProblemData{T}; mode=:nominal) where T
    x, u, _, _, _ = primal_trajectories(data, mode=mode)
    lx = data.objective_data.gradient_state
    lu = data.objective_data.gradient_control
    lxx = data.objective_data.hessian_state_state
    luu = data.objective_data.hessian_control_control
    lux = data.objective_data.hessian_control_state
    objective_gradient!(lx, lu, objectives, x, u)
    objective_hessian!(lxx, luu, lux, objectives, x, u) 
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
    evaluate_derivatives!(problem.objective_data.objectives, problem, mode=mode)
    evaluate_derivatives!(problem.constraints_data, problem, mode=mode)
end
