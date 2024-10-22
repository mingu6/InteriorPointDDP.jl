using JuMP
import Ipopt
using Random
using Plots

Random.seed!(0)
N = 101
h = 0.05
r_car = 0.02
x1 = [0.0; 0.0; 0.0] + rand(3) .* [0.05, 0.05, π / 2]
xN = [1.0; 1.0; π / 4]

nx = 3  # num. state
nu = 2  # num. control

include("../../examples/visualise/concar.jl")

model = Model(
    optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "max_refinement_steps" => 0, "min_refinement_steps" => 0)
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

# ## Dynamics - explicit midpoint for integrator

function car_continuous(x, u)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function car_discrete(x, u)
    x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)
end

# ## constraints

obs_dist(obs_xy) = (x, u) -> begin
    xp = car_discrete(x, u)[1:2]
    xy_diff = xp[1:2]- obs_xy
    return xy_diff' * xy_diff
end
stage_constr_fn = (x, u) -> begin
    # obstacle avoidance constraints w/slack variable,
    # i.e., d_obs^2 - d_thresh^2 >= 0 and d_obs^2 - d_thresh^2 + s = 0, s >= 0
    [(obs[3] + r_car)^2 - obs_dist(obs[1:2])(x, u)
        for (i, obs) in enumerate(xyr_obs)]
end

@constraint(model, x[1, :] == x1)
for k = 1:N-1
    @constraint(model, x[k+1, :] == car_discrete(x[k, :], u[k, :]))
    @constraint(model, ul .<= u[k, :] .<= uu)
    @constraint(model, [0.0, 0.0] .<= x[k+1, 1:2] .<= [1.0, 1.0])
    for (i, obs) in enumerate(xyr_obs)
        @constraint(model, (obs[3] + r_car)^2 - obs_dist(obs[1:2])(x[k, :], u[k, :]) <= 0.0)
    end
end
# @constraint(model, x[N, :] == xN)

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

optimize!(model)

xv = value.(x)
x_sol = [xv[k, :] for k in 1:N]

plot()
plotTrajectory!(x_sol)
for xyr in xyr_obs
    plotCircle!(xyr[1], xyr[2], xyr[3])
end
savefig("plots/concar.png")
