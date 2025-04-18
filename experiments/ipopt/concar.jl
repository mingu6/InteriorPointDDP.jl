using JuMP
import Ipopt
using Random
using Plots
using BenchmarkTools
using Suppressor
using Printf

output = false
benchmark = false
visualise = false
n_benchmark = 10

print_level = output ? 5 : 4

N = 101
Δ = 0.05
r_car = 0.02
xN = [1.0; 1.0; π / 4; 0.0]

visualise && include("../visualise/concar.jl")

nx = 4  # num. state
nu = 2  # num. control
n_ocp = 500

results = Vector{Vector{Any}}()

include("ipopt_parse.jl")


for seed = 1:n_ocp
    Random.seed!(seed)

    model = Model(
        optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6,
            "print_level" => print_level, "print_timing_statistics" => "yes")
        );

    @variable(model, x[1:N, 1:nx]);
    @variable(model, u[1:N-1, 1:nu]);

    # ## control limits

    F_lim = 1.5 + rand()
    τ_lim = 3.0 + 2.0 * rand()

    ul = [-F_lim; -τ_lim]
    uu = [F_lim; τ_lim]

    for t = 1:N-1
        for i = 1:nu
            set_lower_bound(u[t, i], ul[i])
            set_upper_bound(u[t, i], uu[i])
        end
    end

    # ## obstacles

    obs_1 = [0.25, 0.25, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_2 = [0.75, 0.75, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_3 = [0.25, 0.75, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_4 = [0.75, 0.25, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]

    xyr_obs = [obs_1, obs_2, obs_3, obs_4]
    num_obstacles = length(xyr_obs)

    @variable(model, s_obs[1:N-1, 1:num_obstacles] .>= 0.0);  # slacks for obstacle constraints

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

    # ## constraints

    obs_dist(obs_xy) = (x) -> begin
        x2d = x[1:2]
        xy_diff = x2d - obs_xy
        return xy_diff' * xy_diff
    end

    for k = 1:N-1
        @constraint(model, RK4(x[k, :], u[k, :], g) == x[k+1, :])
        for (i, obs) in enumerate(xyr_obs)
            @constraint(model, (obs[3] + r_car)^2 - obs_dist(obs[1:2])(x[k, :]) + s_obs[k, i] == 0.0)
        end
    end

    stage_cost = (x, u) -> begin
        J = 0.0
        # J += Δ * (x - xN)'* (x - xN)
        J += Δ * (u[1:2] .* [5.0, 1.0])' * u[1:2]
        return J
    end

    term_cost = x -> 5e2 * (x - xN)' * (x - xN)

    function cost(x, u)
        J = 0.0
        for k = 1:N-1
            J += stage_cost(x[k, :], u[k, :])
        end
        J += term_cost(x[N, :])
        return J
    end
        
    @objective(model, Min, cost(x, u))

    # ## Plots

    if visualise
        plot(xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), xtickfontsize=14, ytickfontsize=14)
        for xyr in xyr_obs
            plotCircle!(xyr[1], xyr[2], xyr[3])
        end
    end

    set_attribute(model, "print_level", print_level)
    x1 = rand(4) .* [0.0; 0.0; π / 2; 0.0]
    fix.(x[1, :], x1, force = true)
    
    ū = [1.0e-1 .* (rand(2) .- 0.5) for k = 1:N-1]
    
    x̄ = [x1]
    for k in 2:N
        push!(x̄, RK4(x̄[k-1], ū[k-1], g))
    end
    
    for k = 1:N
        for j = 1:nx
            set_start_value(x[k, j], x̄[k][j])
        end
    end
    
    for k = 1:N-1
        for j = 1:nu
            set_start_value(u[k, j], ū[k][j])
        end
    end
    
    for k = 1:N-1
        for j = 1:num_obstacles
            set_start_value(s_obs[k, j], 0.01)
        end
    end
    
    ipopt_out = @capture_out optimize!(model)
    objective, constr_viol, n_iter, succ, _, _ = parse_results_ipopt(ipopt_out)

    # visualise trajectories
    
    xv = value.(x)
    x_sol = [xv[k, :] for k in 1:N]
    visualise && plotTrajectory!(x_sol)
    
    if benchmark
        set_attribute(model, "print_level", 4)
        solver_time_ = 0.0
        wall_time_ = 0.0
        for i = 1:n_benchmark
            ipopt_out = @capture_out optimize!(model)
            _, _, _, _, solver_time, wall_time = parse_results_ipopt(ipopt_out)
            solver_time_ += solver_time
            wall_time_ += wall_time
        end
        solver_time_ /= n_benchmark
        wall_time_ /= n_benchmark
        push!(results, [seed, n_iter, succ, objective, constr_viol, wall_time_, solver_time_])
    else
        push!(results, [seed, n_iter, succ, objective, constr_viol])
    end
    visualise && savefig("plots/concar_IPOPT_$seed.pdf")
end

open("results/concar.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
                            results[i][4], results[i][5], results[i][6], results[i][7])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]), results[i][4], results[i][5])
        end
    end
end
