function gradients!(dynamics::Model, data::ProblemData; mode=:nominal)
    x, u, w = primal_trajectories(data, mode=mode)
    jx = data.model.jacobian_state
    ju = data.model.jacobian_action
    jacobian!(jx, ju, dynamics, x, u, w)
end

function gradients!(costs::Costs, data::ProblemData; mode=:nominal)
    x, u, w = primal_trajectories(data, mode=mode)
    gx = data.costs.gradient_state
    gu = data.costs.gradient_action
    gxx = data.costs.hessian_state_state
    guu = data.costs.hessian_action_action
    gux = data.costs.hessian_action_state
    cost_gradient!(gx, gu, costs, x, u, w)
    cost_hessian!(gxx, guu, gux, costs, x, u, w) 
end

function gradients!(constraints_data::ConstraintsData, problem::ProblemData; mode=:nominal)
    x, u, w = primal_trajectories(problem, mode=mode)
    cx = constraints_data.jacobian_state
    cu = constraints_data.jacobian_action
    jacobian!(cx, cu, constraints_data.constraints, x, u, w)
end

function gradients!(problem::ProblemData; mode=:nominal)
    gradients!(problem.model.dynamics, problem, mode=mode)
    gradients!(problem.costs.costs, problem, mode=mode)
    gradients!(problem.constraints, problem, mode=mode)
end
