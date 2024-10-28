using JuMP
import Ipopt
using Random
using Plots
using BenchmarkTools
using Suppressor
using Printf

visualise = true
output = true
benchmark = true

print_level = output ? 5 : 4

N = 101
h = 0.05
r_car = 0.02
xN = [1.0; 1.0; π / 4]

nx = 3  # num. state
nu = 2  # num. control

include("../../examples/visualise/concar.jl")
include("ipopt_parse.jl")

model = Model(
            optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "max_refinement_steps" => 0, 
                        "min_refinement_steps" => 0, "print_level" => print_level)
            );

@variable(model, x[1:N, 1:nx]);
@variable(model, u[1:N-1, 1:nu]);

# ## control limits

ul = [-0.1; -5.0]
uu = [1.0; 5.0]

# ## obstacles

xyr_obs = [
    [0.05, 0.25, 0.1],
    [0.45, 0.1, 0.15],
    [0.7, 0.7, 0.2],
    [0.35, 0.4, 0.1]
    ]
num_obstacles = length(xyr_obs)

@variable(model, s_obs[1:N-1, 1:num_obstacles]);  # slacks for obstacle constraints

# ## Dynamics - explicit midpoint for integrator

function car_continuous(x, u)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
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
    J += 1e-2 * (x - xN)'* (x - xN)
    J += 1e-1 * (u[1:2] .* [1.0, 0.1])' * u[1:2]
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

open("results/concar.txt", "w") do io
	@printf(io, " seed  iterations  status    objective       primal      time (s)  \n")
	for seed = 1:50
		set_attribute(model, "print_level", print_level)
		Random.seed!(seed)
        x1 = [0.0; 0.0; 0.0] + rand(3) .* [0.05, 0.05, π / 2]
        fix.(x[1, :], x1, force = true)
        
        ū = [1.0e-2 * (rand(2) .- 0.5) for k = 1:N-1]
        
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
		objective = objective_value(model)
		objective, constr_viol, n_iter, succ = parse_results_ipopt(ipopt_out)
		
		if benchmark
            set_attribute(model, "print_level", 0)
            solve_time = @belapsed optimize!($model)
            @printf(io, " %2s     %5s      %5s     %.8f    %.8f    %.5f  \n", seed, n_iter, succ, objective, constr_viol, solve_time)
        else
            @printf(io, " %2s     %5s      %5s     %.8f    %.8f \n", seed, n_iter, succ, objective, constr_viol)
        end
    end
end

if visualise
    xv = value.(x)
    x_sol = [xv[k, :] for k in 1:N]
    
    plot()
    plotTrajectory!(x_sol)
    for xyr in xyr_obs
        plotCircle!(xyr[1], xyr[2], xyr[3])
    end
    savefig("plots/concar.png")
end
