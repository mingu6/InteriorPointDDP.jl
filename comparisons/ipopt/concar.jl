using JuMP
import Ipopt
using Random
using Plots
using BenchmarkTools
using Suppressor
using Printf

visualise = true
output = false
benchmark = true
n_benchmark = 10

print_level = output ? 5 : 4

N = 101
h = 0.05
r_car = 0.02
xN = [1.0; 1.0; π / 4; 0.0]

nx = 4  # num. state
nu = 2  # num. control

include("../../examples/visualise/concar.jl")
include("ipopt_parse.jl")

model = Model(
            optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "max_refinement_steps" => 0, 
                        "min_refinement_steps" => 0, "print_level" => print_level, "print_timing_statistics" => "yes")
            );

@variable(model, x[1:N, 1:nx]);
@variable(model, u[1:N-1, 1:nu]);

# ## control limits

ul = [-2.0; -5.0]
uu = [2.0; 5.0]

# ## obstacles

xyr_obs = [
    [0.05, 0.25, 0.1],
    [0.45, 0.1, 0.15],
    [0.7, 0.7, 0.2],
    [0.30, 0.4, 0.1]
    ]
num_obstacles = length(xyr_obs)

@variable(model, s_obs[1:N-1, 1:num_obstacles]);  # slacks for obstacle constraints

# ## Dynamics - explicit midpoint for integrator

function car_continuous(x, u)
    [x[4] * cos(x[3]); x[4] * sin(x[3]); u[2]; u[1]]
end

function car_discrete(x, u)
    x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)
end

# ## constraints

obs_dist(obs_xy) = (x) -> begin
    xy_diff = x[1:2] - obs_xy
    return xy_diff' * xy_diff
end

for k = 1:N-1
    @constraint(model, x[k+1, :] == car_discrete(x[k, :], u[k, :]))
    @constraint(model, ul .<= u[k, :] .<= uu)
    @constraint(model, [0.0, 0.0] .<= x[k+1, 1:2] .<= [1.0, 1.0])
    @constraint(model, 0.0 .<= s_obs[k, :])
    for (i, obs) in enumerate(xyr_obs)
        @constraint(model, (obs[3] + r_car)^2 - obs_dist(obs[1:2])(x[k+1, :]) + s_obs[k, i] <= 0.0)
    end
end

stage_cost = (x, u) -> begin
    J = 0.0
    J += h * (x - xN)'* (x - xN)
    J += h * (u[1:2] .* [10.0, 1.0])' * u[1:2]
    return J
end

term_cost = x -> 1e3 * (x - xN)' * (x - xN)

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
    plot(xlims=(0, 1), ylims=(0, 1), xtickfontsize=14, ytickfontsize=14)
    for xyr in xyr_obs
        plotCircle!(xyr[1], xyr[2], xyr[3])
    end
end

open("results/concar.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms) \n")
	for seed = 1:50
		set_attribute(model, "print_level", print_level)
		Random.seed!(seed)
        x1 = [0.0; 0.0; 0.0; 0.0] + rand(4) .* [0.05; 0.05; π / 2; 0.0]
        fix.(x[1, :], x1, force = true)
        
        ū = [1.0e-3 * (rand(2) .- 0.5) for k = 1:N-1]
        
        x̄ = [x1]
        for k in 2:N
            push!(x̄, car_discrete(x̄[k-1],  ū[k-1]))
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
		
		if visualise
		    xv = value.(x)
            x_sol = [xv[k, :] for k in 1:N]
            
            plotTrajectory!(x_sol)
        end
		
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
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e    %5.1f        %5.1f  \n", seed, n_iter, succ, objective, constr_viol, wall_time_, solver_time_)
        else
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e \n", seed, n_iter, succ, objective, constr_viol)
        end
    end
end

visualise && savefig("plots/concar_IPOPT.pdf")
