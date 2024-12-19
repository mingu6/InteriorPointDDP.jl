using JuMP
import Ipopt
using Random
using Plots
using MeshCat
using Suppressor
using Printf

visualise = false
output = false
benchmark = false
n_benchmark = 10

print_level = output ? 5 : 4

include("ipopt_parse.jl")
include("../models/cartpole.jl")

if visualise
    include("../visualise/visualise_cartpole.jl")
    !@isdefined(vis) && (vis = Visualizer())
    render(vis)
end

h = 0.05
N = 101

nq = cartpole.nq
nF = cartpole.nu
nx = 2 * nq
nu = nF + nq

xN = [0.0; π; 0.0; 0.0]


model = Model(
            optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "max_refinement_steps" => 0, 
                        "min_refinement_steps" => 0, "print_level" => print_level, "print_timing_statistics" => "yes")
            );

# ## Costs

objt = (x, u) -> h * u[1] * u[1]
objT = (x, u) -> 400. * (x - xN)' * (x - xN)

cost = (x, u) -> begin
	J = 0.0
	for k in 1:N-1
		J += objt(x[k, :], u[k, :])
	end
	J += objT(x[N, :], 0.0)
	return J
end

# ## Dynamics - forward Euler

f = (x, u) -> x + h * [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]  # forward Euler
dyn_con = (x, u) -> implicit_dynamics(cartpole, x, u) * h

# ## Constraints

@variable(model, x[1:N, 1:nx]);
@variable(model, u[1:N-1, 1:nu]);

@objective(model, Min, cost(x, u))

for k = 1:N-1
    @constraint(model, x[k+1, :] == f(x[k, :], u[k, :]))
    @constraint(model, dyn_con(x[k, :], u[k, :]) .== 0.0)
    @constraint(model, 4.0 .>= u[k, 1:nF] .>= -4.0)
end

open("results/cartpole_inverse.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms) \n")
	for seed = 1:50
		set_attribute(model, "print_level", print_level)
		Random.seed!(seed)
		
        # ## Initialise variables and solve
        
        x1 = (rand(4) .- 0.5) .* [0.1, 0.1, 0.1, 0.1]
        fix.(x[1, :], x1, force = true)
        ū = [1.0e-1 * (rand(nu) .- 0.5) for k = 1:N-1]
        
        x̄ = [x1]
        for k in 2:N
            push!(x̄, f(x̄[k-1],  ū[k-1]))
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
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e    %5.1f        %5.1f  \n", seed, n_iter, succ, objective, constr_viol, wall_time_, solver_time_)
        else
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e \n", seed, n_iter, succ, objective, constr_viol)
        end
    end
end


if visualise
    xv = value.(x)
    x_sol = [xv[k, :] for k in 1:N]
	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end
    