function gradients!(dynamics::Model, data::ProblemData{T}; mode=:nominal) where T
    x, u, _ = primal_trajectories(data, mode=mode)
    jx = data.model.jacobian_state
    ju = data.model.jacobian_control
    jacobian!(jx, ju, dynamics, x, u)
end

function gradients!(costs::Costs, data::ProblemData{T}; mode=:nominal) where T
    x, u, _ = primal_trajectories(data, mode=mode)
    gx = data.cost_data.gradient_state
    gu = data.cost_data.gradient_control
    gxx = data.cost_data.hessian_state_state
    guu = data.cost_data.hessian_control_control
    gux = data.cost_data.hessian_control_state
    cost_gradient!(gx, gu, costs, x, u)
    cost_hessian!(gxx, guu, gux, costs, x, u) 
end

function gradients!(constraints_data::ConstraintsData{T}, problem::ProblemData{T}; mode=:nominal) where T
    x, u, _ = primal_trajectories(problem, mode=mode)
    cx = constraints_data.jacobian_state
    cu = constraints_data.jacobian_control
    jacobian!(cx, cu, constraints_data.constraints, x, u)
end

function gradients!(problem::ProblemData{T}; mode=:nominal) where T
    gradients!(problem.model.dynamics, problem, mode=mode)
    gradients!(problem.cost_data.costs, problem, mode=mode)
    gradients!(problem.constr_data, problem, mode=mode)
    vhux = problem.constr_data.vhux
    vhuu = problem.constr_data.vhuu
    constraints = problem.constr_data.constraints
    x, u, _ = primal_trajectories(problem, mode=mode)
    ϕ = mode == :nominal ? problem.nominal_eq_duals : problem.eq_duals
    tensor_contraction!(vhux, vhuu, constraints, x, u, ϕ)
end
