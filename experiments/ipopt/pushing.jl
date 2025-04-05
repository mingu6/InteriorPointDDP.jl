using JuMP
import Ipopt
using Random
using Plots
using BenchmarkTools
using Suppressor
using Printf

output = false
benchmark = false
n_benchmark = 10

print_level = output ? 5 : 4

Δ = 0.04
N = 76

x1 = [0.0, 0.0, 0.0, 0.0]
n_ocp = 100

results = Vector{Vector{Any}}()

include("ipopt_parse.jl")

for seed = 1:n_ocp
    Random.seed!(seed)

    model = Model(
        optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none",
            "print_level" => print_level, "print_timing_statistics" => "yes"
            , "max_resto_iter" => 0, "max_filter_resets" => 0
            )
        );

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

    # xyr_obs = [obs1, obs2, obs3]
    xyr_obs = []
    n_obs = length(xyr_obs)

    nu = 9 + n_obs
    nx = 4

    @variable(model, x[1:N, 1:nx]);
    @variable(model, u[1:N-1, 1:nu]);

    # ## control limits

    ul = [[0.0, -force_lim, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, -0.9]; zeros(n_obs)]
    uu = [[force_lim, force_lim, vel_lim, vel_lim, Inf, Inf, Inf, Inf, 0.9]; Inf .* ones(n_obs)]

    for t = 1:N-1
        for i = 1:nu
            if !isinf(ul[i])
                set_lower_bound(u[t, i], ul[i])
            end
            if !isinf(uu[i])
                set_upper_bound(u[t, i], uu[i])
            end
        end
    end

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

    obs_dist(obs_xy) = (x, u) -> begin
        xp = x[1:2]
        xy_diff = xp - obs_xy
        return xy_diff' * xy_diff
    end

    function constr(x, u)
        [
        μ_fric * u[1] - u[2] - u[5];
        μ_fric * u[1] + u[2] - u[6];
        u[5] * u[3] + u[7];
        u[6] * u[4] + u[8];
        f(x, u)[4] - u[9]  # bound constraint on ϕ_t
        # [(obs[3] + r_push)^2 - obs_dist(obs[1:2])(x, u) + u[9 + i]
        #     for (i, obs) in enumerate(xyr_obs)]
        ]
    end

    for k = 1:N-1
        @constraint(model, x[k+1, :] == f(x[k, :], u[k, :]))
        @constraint(model, constr(x[k, :], u[k, :]) == zeros(5 + n_obs))
    end

    # objective

    stage_cost = (x, u) -> begin
        J = 0.0
        J += 1e-2 * u[1:2]' * u[1:2] + 50. * u[7:8]' * u[7:8]
        return J
    end

    term_cost = x -> 10. * (x - xN)' * (x - xN)

    function cost(x, u)
        J = 0.0
        for k = 1:N-1
            J += stage_cost(x[k, :], u[k, :])
        end
        J += term_cost(x[N, :])
        return J
    end
        
    @objective(model, Min, cost(x, u))

    set_attribute(model, "print_level", print_level)
    fix.(x[1, :], x1, force = true)
    
    xs_init = LinRange(x1, xN, N)[2:end]
    ū = [[0.1 * rand(); 0.1 * (rand() - 0.5); 1e-2 .* ones(7 + n_obs)] for k = 1:N-1]
    
    x̄ = [x1]
    for k in 2:N
        push!(x̄, f(x̄[k-1], ū[k-1]))
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
    
    ipopt_out = @capture_out optimize!(model)
    objective, constr_viol, n_iter, succ, _, _ = parse_results_ipopt(ipopt_out)

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
end

open("results/pushing.txt", "w") do io
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
