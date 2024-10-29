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

include("ipopt_parse.jl")

h = 0.01
N = 101
xN = [1.0; 0.0]
x1 = [0.0; 0.0]

nx = 2  # num. state
nu = 3  # num. control

model = Model(
            optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "max_refinement_steps" => 0, 
                        "min_refinement_steps" => 0, "print_level" => print_level)
            );

function blockmove_continuous(x, u)
    return [x[2], u[1]]
end
blockmove_discrete = (x, u) -> x + h * blockmove_continuous(x + 0.5 * h * blockmove_continuous(x, u), u)

function cost(x, u)
    J = 0.0
    for k = 1:N-1
        J += h * (u[k, 2] + u[k, 3])
    end
    J += 400.0 * (x[N, :] - xN)' * (x[N, :] - xN)
    return J
end

@variable(model, x[1:N, 1:nx]);
@variable(model, u[1:N-1, 1:nu]);

@objective(model, Min, cost(x, u))

fix.(x[1, :], x1, force = true)
for k = 1:N-1
    @constraint(model, x[k+1, :] == blockmove_discrete(x[k, :], u[k, :]))
    @constraint(model, u[k, 2] - u[k, 3] == u[k, 1] * x[k, 2])
    @constraint(model, -10.0 <= u[k, 1] <= 10.0)
    @constraint(model, u[k, 2:3] .>= 0.0)
end

open("results/blockmove.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        time (s)  \n")
	for seed = 1:50
		set_attribute(model, "print_level", print_level)
		Random.seed!(seed)
		
        # ## Initialise variables and solve
        
        ū = [[1.0e-0 * (randn(1) .- 0.5); 0.01 * ones(2)] for k = 1:N-1]
        x̄ = [x1]
        for k in 2:N
            push!(x̄, blockmove_discrete(x̄[k-1],  ū[k-1]))
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
		objective = objective_value(model)
		objective, constr_viol, n_iter, succ = parse_results_ipopt(ipopt_out)
		
		if benchmark
            set_attribute(model, "print_level", 0)
            solve_time = @belapsed optimize!($model)
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e    %.5f  \n", seed, n_iter, succ, objective, constr_viol, solve_time)
        else
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e \n", seed, n_iter, succ, objective, constr_viol)
        end
    end
end

# ## Plot solution

if visualise
    xv = value.(x)
	x_sol = [xv[k, :] for k in 1:N]
	uv = value.(u)
	u_sol = [uv[k, :] for k in 1:N-1]
    
    x = map(x -> x[1], x_sol)
    v = map(x -> x[2], x_sol)
    u = [map(u -> u[1], u_sol); 0.0]
    work = [abs(vk * uk) for (vk, uk) in zip(v, u)]
    plot(range(0, (N-1) * h, length=N), [x v u work], label=["x" "v" "u" "work"])
    savefig("plots/blockmove.png")
    
    println("Total absolute work: ", sum(work))
end
