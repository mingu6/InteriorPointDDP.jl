function gradients!(dynamics::Model, data::ProblemData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    jx = data.model.jacobian_state
    ju = data.model.jacobian_action
    jacobian!(jx, ju, dynamics, x, u, w)
end

function gradients!(objective::Objective, data::ProblemData; mode=:nominal)
    x, u, w = trajectories(data, mode=mode)
    gx = data.objective.gradient_state
    gu = data.objective.gradient_action
    gxx = data.objective.hessian_state_state
    guu = data.objective.hessian_action_action
    gux = data.objective.hessian_action_state
    cost_gradient!(gx, gu, objective, x, u, w)
    cost_hessian!(gxx, guu, gux, objective, x, u, w) 
end

function gradients!(constraints_data::ConstraintsData, problem::ProblemData; mode=:nominal)
    x, u, w = trajectories(problem, mode=mode)
    cx = constraints_data.jacobian_state
    cu = constraints_data.jacobian_action
    jacobian!(cx, cu, constraints_data.constraints, x, u, w)
end

function gradients!(problem::ProblemData; mode=:nominal)
    gradients!(problem.model.dynamics, problem, mode=mode)
    gradients!(problem.objective.costs, problem, mode=mode)
    gradients!(problem.constraints, problem, mode=mode)
end
