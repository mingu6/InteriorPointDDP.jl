function gradients!(dynamics::Model, data::ProblemData; mode=:nominal)
    x, u, _ = primal_trajectories(data, mode=mode)
    jx = data.model.jacobian_state
    ju = data.model.jacobian_action
    jacobian!(jx, ju, dynamics, x, u)
end

function gradients!(costs::Costs, data::ProblemData; mode=:nominal)
    x, u, _ = primal_trajectories(data, mode=mode)
    gx = data.cost_data.gradient_state
    gu = data.cost_data.gradient_action
    gxx = data.cost_data.hessian_state_state
    guu = data.cost_data.hessian_action_action
    gux = data.cost_data.hessian_action_state
    cost_gradient!(gx, gu, costs, x, u)
    cost_hessian!(gxx, guu, gux, costs, x, u) 
end

function gradients!(constraints_data::ConstraintsData, problem::ProblemData; mode=:nominal)
    x, u, _ = primal_trajectories(problem, mode=mode)
    cx = constraints_data.jacobian_state
    cu = constraints_data.jacobian_action
    jacobian!(cx, cu, constraints_data.constraints, x, u)
end

function gradients!(problem::ProblemData; mode=:nominal)
    gradients!(problem.model.dynamics, problem, mode=mode)
    gradients!(problem.cost_data.costs, problem, mode=mode)
    gradients!(problem.constr_data, problem, mode=mode)
end
