using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using BenchmarkTools
using Printf

benchmark = false
verbose = true

Δ = 0.04
N = 76
n_ocp = 500

x1 = [0.0, 0.0, 0.0, 0.0]

options = Options()
options.scaling_penalty = 1.3
options.initial_constraint_penalty = 2e-0
options.max_dual_updates = 20

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
	Random.seed!(seed)

    x1 = [0.0, 0.0, 0.0, 0.0]

    xl = 0.07 + (rand() - 0.5) * 0.02
    xr = 0.12 + (rand() - 0.5) * 0.03
    c = 0.03711 + 0.01 * (rand() - 0.5)  # ellipsoidal approximation ratio

    xN = [0.3, 0.4, 1.5 * pi, 0.0]
    μ_fric = 0.2 + 0.05 * (rand() - 0.5) # friction coefficient b/w pusher and slider
    force_lim = 0.3 + 0.1 * (rand() - 0.5)
    vel_lim = 3.0 + rand()
    r_push = 0.01 + 0.005 * (rand() - 0.5)

    nu = 8
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

    stage_fn = (x, u) -> 1e-2 * dot(u[1:2], u[1:2]) + 50. * dot(u[7:8], u[7:8])
    term_fn = (x, u) -> 10.0 * dot(x - xN, x - xN)

    stage_objective = Cost(stage_fn, nx, nu)
    term_objective = Cost(term_fn, nx, 0)
    objective = [
        [stage_objective for k = 1:N-1]...,
        term_objective,
    ]

    function eval_objective(x, u)
        J = 0.0
        for t = 1:N-1
            J += stage_fn(x[t], u[t])
        end
        J += term_fn(x[N], 0.0)
    end

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

        -u[1];
        u[1] - force_lim;
        -u[2] - force_lim;
        u[2] - force_lim;
        -u[3];
        u[3] - vel_lim;
        -u[4];
        u[4] - vel_lim;
        -u[5];
        -u[6];
        x[4] - 0.9;
        -x[4] - 0.9
        ]
    end

    path_constr = Constraint(constr, nx, nu, indices_inequality=collect(5:16))
    constraints = [[path_constr for k = 1:N-1]..., Constraint()]

    function eval_constraints_infnorm(x, u)
        θ = 0.0
        for t in 1:N-1
            h = constr(x[t], u[t])
            θ = max(θ, norm(max.(h[5:16], zeros(12)), Inf))
            θ = max(θ, norm(h[1:4], Inf))
        end
        return θ
    end

    # Bounds

    solver = Solver(dynamics, objective, constraints; options=options)
    solver.options.verbose = verbose
        
    # ## Initialise solver and solve
    
    ū = [[0.1 * rand(); 0.1 * (rand() - 0.5); 1e-2 .* ones(6)] for k = 1:N-1]
    x̄ = rollout(dynamics, x1, ū)
    solve!(solver, x̄, ū)

    x_sol, u_sol = get_trajectory(solver)

    J = eval_objective(x_sol, u_sol)
    θ = eval_constraints_infnorm(x_sol, u_sol)

    if benchmark
        solver.options.verbose = false
        solve_time = @belapsed solve!($solver, $x̄, $ū) samples=10 setup=(solver=Solver($dynamics, $objective, $constraints; options=$options))
        push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ, solve_time, 0.0])
    else
        push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ])
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

