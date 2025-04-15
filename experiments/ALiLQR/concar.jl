using IterativeLQR 
using LinearAlgebra
using Plots
using Random
using BenchmarkTools
using Printf

benchmark = true
verbose = true
visualise = false

N = 101
Δ = 0.05
r_car = 0.02

options = Options()
options.scaling_penalty = 1.7
options.initial_constraint_penalty = 5e-4
options.max_dual_updates = 300
options.constraint_tolerance = 1e-6

# ## car 
num_state = 4
num_control = 2
n_ocp = 500

visualise && include("../visualise/concar.jl")

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    xN = [1.0; 1.0; π / 4; 0.0]

    # ## control limits
    F_lim = 1.5 + rand()
    τ_lim = 3.0 + 2.0 * rand()

    ul = [-F_lim; -τ_lim]
    uu = [F_lim; τ_lim]

    # ## obstacles
    obs_1 = [0.25, 0.25, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_2 = [0.75, 0.75, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_3 = [0.25, 0.75, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_4 = [0.75, 0.25, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]

    xyr_obs = [obs_1, obs_2, obs_3, obs_4]
    num_obstacles = length(xyr_obs)
    num_primal = num_control #+ num_obstacles

    # ## Dynamics - RK4

    # continuous time dynamics
    function g(x, u)
        [x[4] * cos(x[3]); x[4] * sin(x[3]); u[2]; u[1]]
    end

    function RK4(x, u, g)
        k1 = g(x, u)
        k2 = g(x + Δ * 0.5 * k1, u)
        k3 = g(x + Δ * 0.5 * k2, u)
        k4 = g(x + Δ * k3, u)
        return x + Δ / 6 * (k1 + k2 + k3 + k4)
    end

    f = (x, u) -> RK4(x, u, g)

    car = Dynamics(f, num_state, num_primal)
    dynamics = [car for k = 1:N-1] 

    # ## objective 

    stage_cost = (x, u) -> begin
        J = 0.0
        J += Δ * dot(x - xN, x - xN)
        J += Δ * dot(u[1:2] .* [5.0, 1.0], u[1:2])
        return J
    end
    term_cost = (x, u) -> 5e2 * dot(x - xN, x - xN)
    objective = [
        [Cost(stage_cost, num_state, num_primal) for k = 1:N-1]...,
        Cost(term_cost, num_state, 0)
    ]

    function eval_objective(x, u)
        J = 0.0
        for t = 1:N-1
            J += stage_cost(x[t], u[t])
        end
        J += term_cost(x[N], 0.0)
    end

    # ## constraints - waypoints are constraints for iLQR

    obs_dist(obs_xy) = (x, u) -> begin
        x2d = x[1:2]
        xy_diff = x2d[1:2] - obs_xy
        return dot(xy_diff, xy_diff)
    end
    path_constr_fn = (x, u) -> begin
    [
        ul - u[1:num_control]; ## control limit (lower)
        u[1:num_control] - uu; ## control limit (upper)
        # # slack variables for obstacle
        # -u[num_control .+ (1:num_obstacles)];
        # # obstacle avoidance constraints i.e., d_thresh^2 - d_obs^2 <= 0 
        [(obs[3] + r_car)^2 - obs_dist(obs[1:2])(x, u) #+ u[num_control + i]
            for (i, obs) in enumerate(xyr_obs)]
    ]
    end

    num_ineq = 2*num_control+num_obstacles
    obs_constr = Constraint(path_constr_fn, num_state, num_primal,
        indices_inequality=collect(1:num_ineq))

    constraints = [[obs_constr for k = 1:N-1]..., Constraint()]

    function eval_constraints_infnorm(x, u)
        θ = 0.0
        for t in 1:N-1
            h = path_constr_fn(x[t], u[t])
            θ += norm(max.(h[1:num_ineq], zeros(num_ineq)), Inf)
        end
        return θ
    end

    # ## Plots

    if visualise
        plot(xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), xtickfontsize=14, ytickfontsize=14)
        for xyr in xyr_obs
            plotCircle!(xyr[1], xyr[2], xyr[3])
        end
    end
    
    # ## Initialise solver and solve

    solver = Solver(dynamics, objective, constraints; options=options)
    solver.options.verbose = verbose
    
    x1 = rand(num_state) .* [0.0; 0.0; π / 2; 0.0]
    ū = [1.0e-1 .* (rand(num_control) .- 0.5) for k = 1:N-1]
    x̄ = rollout(dynamics, x1, ū)
    
    solve!(solver, x̄, ū)
    
    x_sol, u_sol = get_trajectory(solver)
    visualise && plotTrajectory!(x_sol)

    J = eval_objective(x_sol, u_sol)
    θ = eval_constraints_infnorm(x_sol, u_sol)
    
    if benchmark
        solver.options.verbose = false
        solve_time = @belapsed solve!($solver, $x̄, $ū) samples=10 setup=(solver=Solver($dynamics, $objective, $constraints; options=$options))
        push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ, solve_time, 0.0])
    else
        push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ])
    end
    visualise && savefig("plots/concar_ALiLQR_$seed.pdf")
end

open("results/concar.txt", "w") do io
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
