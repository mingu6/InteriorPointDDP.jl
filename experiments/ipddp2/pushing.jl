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

Δ = 0.04
N = 76
n_ocp = 100

x1 = [0.0, 0.0, 0.0, 0.0]
options = Options{Float64}(verbose=false)

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
	Random.seed!(seed)

    xl = 0.07 + (rand() - 0.5) * 0.02
    xr = 0.12 + (rand() - 0.5) * 0.03
    c = 0.03711 + 0.01 * (rand() - 0.5)  # ellipsoidal approximation ratio

    xN = [0.3, 0.4, 1.5 * pi, 0.0]
    μ_fric = 0.2 + 0.05 * (rand() - 0.5) # friction coefficient b/w pusher and slider
    force_lim = 0.3 + 0.1 * (rand() - 0.5)
    vel_lim = 3.0 + rand()
    r_push = 0.01 + 0.005 * (rand() - 0.5)

    # obs1 = [0.2, 0.2, 0.05] + [0.05 * (rand() - 0.5), 0.05 * (rand() - 0.5), 0.005 * (rand() - 0.5)]
    # obs2 = [0.0, 0.4, 0.05] + [0.025 * rand(), 0.05 * (rand() - 0.5), 0.005 *  (rand() - 0.5)]
    # obs3 = [0.3, 0.0, 0.05] + [0.05 * (rand() - 0.5), 0.025 * rand(), 0.005 *  (rand() - 0.5)]

    # obs1 = [0.2, 0.2, 0.05] + [0.05 * (rand() - 0.5), 0.05 * (rand() - 0.5), 0.005 * (rand() - 0.5)]
    # obs2 = [0.1, 0.5, 0.05] + [0.025 * rand(), 0.05 * (rand() - 0.5), 0.005 *  (rand() - 0.5)]

    # xyr_obs = [obs1, obs2, obs3]
    # xyr_obs = [obs1, obs2]
    xyr_obs = []
    n_obs = length(xyr_obs)

    nu = 9 + n_obs
    nx = 4

    # dynamics

    function R(θ)
        return [[cos(θ); sin(θ); 0] [-sin(θ); cos(θ); 0] [0.0; 0.0; 1.0]]
    end

    L = [1.0; 1.0; c^(-2)]

    function Jc(ϕ)
        return [[1.0; 0.0] [0.0; 1.0] [xl / 2 * tan(ϕ); -xl / 2]]
    end

    function fc(x, u)
        θ = x[3]
        ϕ = x[4]
        return [R(θ) * (L .* (transpose(Jc(ϕ)) * u[1:2])); u[3] - u[4]]
    end

    function f(x, u)
        return x + Δ .* fc(x, u)
    end

    dynamics = [Dynamics(f, nx, nu) for k = 1:N-1]

    # objective

    stage_objective = Objective((x, u) -> 1e-2 * dot(u[1:2], u[1:2]) + 50. * dot(u[7:8], u[7:8]), nx, nu)
    objective = [
        [stage_objective for k = 1:N-1]...,
        Objective((x, u) -> 10.0 * dot(x - xN, x - xN), nx, 0),
    ]

    # constraints

    obs_dist(obs_xy) = (x, u) -> begin
        xp = f(x, u)[1:2]
        xy_diff = xp[1:2] - obs_xy
        return dot(xy_diff, xy_diff)
    end

    function constr(x, u)
        [
        μ_fric * u[1] - u[2] - u[5];
        μ_fric * u[1] + u[2] - u[6];
        u[5] * u[3] + u[7];
        u[6] * u[4] + u[8];
        f(x, u)[4] - u[9];  # bound constraint on ϕ_t
        [(obs[3] + r_push)^2 - obs_dist(obs[1:2])(x, u) + u[9 + i]
            for (i, obs) in enumerate(xyr_obs)]
        ]
    end

    path_constr = Constraint(constr, nx, nu)
    constraints = [path_constr for k = 1:N-1]

    # Bounds

    bound = Bound([[0.0, -force_lim, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -0.9]; zeros(n_obs)],
                [[force_lim, force_lim, vel_lim, vel_lim, Inf, Inf, Inf, Inf, 0.9]; Inf .* ones(n_obs)])
    bounds = [bound for k = 1:N-1]

    solver = Solver(Float64, dynamics, objective, constraints, bounds, options=options)
    solver.options.verbose = verbose
        
    # ## Initialise solver and solve
    
    ū = [[0.1 * rand(); 0.1 * (rand() - 0.5); 1e-2 .* ones(7 + n_obs)] for k = 1:N-1]
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
end

open("results/pushing.txt", "w") do io
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

