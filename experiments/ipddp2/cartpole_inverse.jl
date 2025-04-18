using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf

visualise = false
benchmark = false
verbose = true
n_benchmark = 10

T = Float64
Δ = 0.05
N = 101
n_ocp = 500

include("../models/cartpole.jl")

if visualise
	include("../visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

xN = T[0.0; π; 0.0; 0.0]

options = Options{T}(verbose=verbose)

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    cartpole = Cartpole{T}(2, 1,
        T(0.8) + T(0.4) * rand(T),
        T(0.2) + T(0.2) * rand(T),
        T(0.4) + T(0.2) * rand(T),
        T(9.81))

    nq = cartpole.nq
    nF = cartpole.nu
    nx = 2 * nq
    nu = nF + nq  # torque and acceleration now decision variables/"controls"

    # ## Dynamics - forward Euler

    f = (x, u) -> x + Δ * [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]
    cartpole_dyn = Dynamics(f, nx, nu)
    dynamics = [cartpole_dyn for k = 1:N-1]

    # ## Objective

    stage = Objective((x, u) -> 0.1 * Δ * dot(u[1], u[1]), nx, nu)
    objective = [
        [stage for k = 1:N-1]...,
        Objective((x, u) -> 100.0 * dot(x - xN, x - xN), nx, 0)
    ] 

    # ## Constraints

    path_constr = Constraint((x, u) -> implicit_dynamics(cartpole, x, u) * Δ, nx, nu)

    constraints = [path_constr for k = 1:N-1]

    # ## Bounds

    limit = T(2.0) * rand(T) + T(4.0)  # bound is in [4, 6]
    bound = Bound(
        [-limit * ones(T, nF); -T(Inf) * ones(T, nq)],
        [limit * ones(T, nF); T(Inf) * ones(T, nq)]
    )
    bounds = [bound for k in 1:N-1]

    solver = Solver(T, dynamics, objective, constraints, bounds, options=options)
    solver.options.verbose = verbose
    
    # ## Initialise solver and solve
    
    x1 = (rand(T, 4) .- T(0.5)) .* T[0.05, 0.05, 0.05, 0.05]
    ū = [T(1.0e-2) * (rand(T, nu) .- T(0.5)) for k = 1:N-1]
    solve!(solver, x1, ū)

    if benchmark
        solver.options.verbose = false
        solver_time = 0.0
        wall_time = 0.0
        for i in 1:n_benchmark
            solve!(solver, x1, ū)
            solver_time += solver.data.solver_time
            wall_time += solver.data.wall_time
        end
        solver_time /= n_benchmark
        wall_time /= n_benchmark
        push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time, solver_time])
    else
        push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf])
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
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
                            results[i][4], results[i][5], results[i][6] * 1000, results[i][7] * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0, results[i][4], results[i][5])
        end
    end
end
