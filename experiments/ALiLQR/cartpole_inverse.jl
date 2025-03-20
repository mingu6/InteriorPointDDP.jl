using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using MeshCat
using BenchmarkTools
using Printf

visualise = false
benchmark = false
verbose = true

Δ = 0.05
N = 101
n_ocp = 500

options = Options()
options.scaling_penalty = 1.3
options.initial_constraint_penalty = 1e-3
options.max_dual_updates = 300
options.constraint_tolerance = 1e-10

include("../models/cartpole.jl")

if visualise
	include("../visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

xN = [0.0; π; 0.0; 0.0]

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    cartpole = Cartpole{Float64}(2, 1,
        0.8 + 0.4 * rand(),
        0.2 + 0.2 * rand(),
        0.4 + 0.2 * rand(),
        9.81)

    nq = cartpole.nq
    nF = cartpole.nu
    nx = 2 * nq
    nu = nF + nq  # torque and acceleration now decision variables/"controls"

    # ## Dynamics - forward Euler

    f = (x, u) -> x + Δ * [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]  # forward Euler
    cartpole_dyn = Dynamics(f, nx, nu)
    dynamics = [cartpole_dyn for k = 1:N-1]

    # ## Costs

    stage_cost = (x, u) -> 0.1 * Δ * dot(u[1], u[1])
    stage = Cost(stage_cost, nx, nu)
    term_cost = (x, u) -> 100. * dot(x - xN, x - xN)
    objective = [[stage for k = 1:N-1]..., Cost(term_cost, nx, 0)]

    function eval_objective(x, u)
        J = 0.0
        for t = 1:N-1
            J += stage_cost(x[t], u[t])
        end
        J += term_cost(x[N], 0.0)
    end

    # ## Constraints

    limit = 2.0 * rand() + 4.0  # bound is in [4, 6]
    path_constr_fn = (x, u) -> [
                        u[1] - limit;
                        -u[1] - limit;
                        Δ * implicit_dynamics(cartpole, x, u)
                        ]

    path_constr = Constraint(path_constr_fn, nx, nu, indices_inequality=collect(1:2))
    constraints = [[path_constr for k = 1:N-1]..., Constraint()]

    function eval_constraints_1norm(x, u)
        θ = 0.0
        for t in 1:N-1
            h = path_constr_fn(x[t], u[t])
            θ += norm(max.(h[1:2], zeros(2)), 1)
            θ += norm(h[3:end], 1)
        end
        return θ
    end

    # ## Initialise solver
            
    solver = Solver(dynamics, objective, constraints; options=options)

    x1 = (rand(4) .- 0.5) .* [0.05, 0.05, 0.05, 0.05]
    ū = [1.0e-2 * (rand(nu) .- 0.5) for k = 1:N-1]
    x̄ = rollout(dynamics, x1, ū)
    
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

    # ## Visualise solution

    if visualise && seed == 1
        x_sol, u_sol = get_trajectory(solver)

        q_sol = [x[1:nq] for x in x_sol]
        visualize!(vis, cartpole, q_sol, Δt=Δ);
    end
end

open("results/cartpole_inverse.txt", "w") do io
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
