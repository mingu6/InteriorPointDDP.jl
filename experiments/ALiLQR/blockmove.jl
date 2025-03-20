using IterativeLQR 
using LinearAlgebra
using Random
using BenchmarkTools
using Printf

benchmark = false
verbose = true

N = 101
Δ = 0.01
x0 = [0.0; 0.0]

options = Options()
options.constraint_tolerance = 1e-8

num_state = 2
num_control = 3  # force and two slack variables to represent abs work
n_ocp = 500

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    # ## Dynamics - Forward Euler

    f = (x, u) -> x + Δ * [x[2], u[1]]  # forward Euler

    blockmove_dyn = Dynamics(f, num_state, num_control)
    dynamics = [blockmove_dyn for k = 1:N-1]

    # ## Costs

    xN_y = 1.0 + rand() * 0.2
    xN_v = (rand() - 0.5) * 0.2
    xN = [xN_y; xN_v]
    stage_cost = (x, u) -> Δ * (u[2] + u[3])
    stage = Cost(stage_cost, num_state, num_control)
    term_cost = (x, u) -> 400.0 * dot(x - xN, x - xN)
    objective = [[stage for k = 1:N-1]..., Cost(term_cost, num_state, 0)]

    function eval_objective(x, u)
        J = 0.0
        for t = 1:N-1
            J += stage_cost(x[t], u[t])
        end
        J += term_cost(x[N], 0.0)
    end

    # ## Constraints

    limit = 5.0 * rand() + 5.0  # bound is in [5, 10]
    path_constr_fn = (x, u) -> [
        u[2] - u[3] - u[1] * x[2],
        - u[1] - limit,
        -u[2],
        -u[3],
        u[1] - limit
        ]
    path_constr = Constraint(path_constr_fn, num_state, num_control, indices_inequality=collect(2:4))

    constraints = [[path_constr for k = 1:N-1]..., Constraint()]

    function eval_constraints_1norm(x, u)
        θ = 0.0
        for t in 1:N-1
            h = path_constr_fn(x[t], u[t])
            θ += norm(max.(h[2:end], zeros(4)), 1)
            θ += abs(h[1])
        end
        return θ
    end

    ū = [[1.0e-0 * (rand(1) .- 0.5); -0.01 * ones(2)] for k = 1:N-1]
    x̄ = rollout(dynamics, x0, ū)

    solver = Solver(dynamics, objective, constraints; options=options)
    solver.options.verbose = verbose
    
    solve!(solver, x̄, ū)

    x_sol, u_sol = get_trajectory(solver)

    J = eval_objective(x_sol, u_sol)
    θ = eval_constraints_1norm(x_sol, u_sol)
    
    if benchmark
        solver.options.verbose = false
        solve_time = @belapsed solve!($solver, $x̄, $ū) samples=10 setup=(solver=Solver($dynamics, $objective, $constraints; options=$options))
        push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ, solve_time * 1000, 0.0])
    else
        push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ])
    end
end

open("results/blockmove.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
                            results[i][4], results[i][5], results[i][6] * 1000, results[i][7] * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]), results[i][4], results[i][5])
        end
    end
end
