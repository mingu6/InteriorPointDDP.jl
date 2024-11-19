using JuMP
import Ipopt
using Random
using Plots
using MeshCat
using Suppressor
using Printf

visualise = false
output = false
benchmark = true
n_benchmark = 10

print_level = output ? 5 : 4

include("ipopt_parse.jl")
include("../../examples/models/cartpole.jl")

if visualise
    include("../../examples/visualise/visualise_cartpole.jl")
    !@isdefined(vis) && (vis = Visualizer())
    render(vis)
end

h = 0.05
N = 101

nq = cartpole.nq
nu = cartpole.nu
nx = 2 * nq
ny = nu + nq

xN = [0.0; π; 0.0; 0.0]


model = Model(
            optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "max_refinement_steps" => 0, 
                        "min_refinement_steps" => 0, "print_level" => print_level, "print_timing_statistics" => "yes")
            );

# ## Costs

objt = (x, y) -> h * y[1] * y[1]
objT = (x, y) -> 400. * (x - xN)' * (x - xN)

cost = (x, u) -> begin
	J = 0.0
	for k in 1:N-1
		J += objt(x[k, :], u[k, :])
	end
	J += objT(x[N, :], 0.0)
	return J
end

# ## Dynamics - implicit dynamics with RK2 integration

f = (x, y) -> [x[nq .+ (1:nq)]; y[nu .+ (1:nq)]]
cartpole_discrete = (x, y) -> x + h * f(x + 0.5 * h * f(x, y), y)  # Explicit midpoint
dyn_con = (x, y) -> implicit_dynamics(cartpole, x, y) * h

# ## Constraints

@variable(model, x[1:N, 1:nx]);
@variable(model, y[1:N-1, 1:ny]);

@objective(model, Min, cost(x, y))

for k = 1:N-1
    @constraint(model, x[k+1, :] == cartpole_discrete(x[k, :], y[k, :]))
    @constraint(model, dyn_con(x[k, :], y[k, :]) .== 0.0)
    @constraint(model, 4.0 .>= y[k, 1:nu] .>= -4.0)
end

open("results/cartpole.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms) \n")
	for seed = 1:50
		set_attribute(model, "print_level", print_level)
		Random.seed!(seed)
		
        # ## Initialise variables and solve
        
        x1 = [0.0; 0.0; 0.0; 0.0] + (rand(4) .- 0.5) .* [0.05, 0.2, 0.1, 0.1]
        fix.(x[1, :], x1, force = true)
        ȳ = [1.0e-2 * (rand(ny) .- 0.5) for k = 1:N-1]
        
        x̄ = [x1]
        for k in 2:N
            push!(x̄, cartpole_discrete(x̄[k-1],  ȳ[k-1]))
        end
        
        for k = 1:N
            for j = 1:nx
                set_start_value(x[k, j], x̄[k][j])
            end
        end
        
        for k = 1:N-1
            for j = 1:ny
                set_start_value(y[k, j], ȳ[k][j])
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
    yv = value.(y)
    u_sol = [yv[k, 1:nu] for k in 1:N-1]
    
	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end
    