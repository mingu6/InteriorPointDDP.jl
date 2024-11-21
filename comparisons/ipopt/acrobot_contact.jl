using JuMP
import Ipopt
using Random
using Plots
using MeshCat
using Suppressor
using Printf
using LaTeXStrings

visualise = true
output = false
benchmark = true
n_benchmark = 10

print_level = output ? 5 : 4

h = 0.05
N = 101

include("../../examples/models/acrobot.jl")
include("ipopt_parse.jl")

if visualise
	include("../../examples/visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = acrobot_impact.nq
nc = acrobot_impact.nc
nF = acrobot_impact.nu
nx = 2 * nq
nu = nF + nq + 2 * nc

q1 = [0.0; 0.0]
q2 = [0.0; 0.0]
x1 = [q1; q2]
qN = [π; 0.0]
xN = [qN; qN]

model = Model(
            optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_method" => "none", "max_refinement_steps" => 0, 
                        "min_refinement_steps" => 0, "print_level" => print_level, "print_timing_statistics" => "yes")
            );

acrobot_discrete = (x, u) -> [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]

# ## Costs

function objt(x, u)
	J = 0.01 * h * u[1] * u[1]
	return J
end

function objT(x, u)
	J = 0.0 
	
	q1 = x[1:acrobot_impact.nq] 
	q2 = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
	v1 = (q2 - q1) ./ h

	J += 200.0 *  v1' * v1
    J += 500.0 * (q2 - qN)' * (q2 - qN)
	return J
end

cost = (x, u) -> begin
	J = 0.0
	for k in 1:N-1
		J += objt(x[k, :], u[k, :])
	end
	J += objT(x[N, :], 0.0)
end

# ## Constraints

constr = (x, u) -> implicit_contact_dynamics(acrobot_impact, x, u, h, 1e-4)

@variable(model, x[1:N, 1:nx]);
@variable(model, u[1:N-1, 1:nu]);

@objective(model, Min, cost(x, u))

fix.(x[1, :], x1, force = true)
for k = 1:N-1
    @constraint(model, x[k+1, :] == acrobot_discrete(x[k, :], u[k, :]))
    @constraint(model, constr(x[k, :], u[k, :]) .== 0.0)
    @constraint(model, u[k, (nF + nq) .+ (1:2*nc)] .>= 0.0)
    @constraint(model, 10.0 .>= u[k, 1] .>= -10.0)
end

open("results/acrobot_contact.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  solver (ms) \n")
	for seed = 1:50
		set_attribute(model, "print_level", print_level)
		Random.seed!(seed)
		
		# ## Initialise variables and solve
		
		q2_init = LinRange(q1, qN, N)[2:end]
		ū = [[1.0e-2 * (rand(nF) .- 0.5); q2_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
		
		x̄ = [x1]
		for k in 2:N
		    push!(x̄, acrobot_discrete(x̄[k-1],  ū[k-1]))
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
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e    %5.1f         %5.1f  \n", seed, n_iter, succ, objective, constr_viol, wall_time_, solver_time_)
        else
            @printf(io, " %2s     %5s      %5s     %.8e    %.8e \n", seed, n_iter, succ, objective, constr_viol)
        end
	end
end

if visualise
	xv = value.(x)
	x_sol = [xv[k, :] for k in 1:N]
	uv = value.(u)
	u_sol = [uv[k, :] for k in 1:N-1]
	
	q_e = map(u -> u[nF + 2], u_sol)
	ϕ1 = map(qe -> π / 2 - qe, q_e)
	ϕ2 = map(qe -> qe + π / 2, q_e)
	λ1 = map(x -> x[end-1], u_sol)
	λ2 = map(x -> x[end], u_sol)
	u = map(x -> x[1], u_sol)
	plot(range(0, h * (N-1), N-1), [ϕ1 ϕ2 λ1 λ2], xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylims=(0,5),
		legendfontsize=12, linewidth=2, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], 
		background_color_legend = nothing, label=[L"$\phi(q)_1$" L"$\phi(q)_2$" L"$\lambda_1$" L"$\lambda_2$"])
	savefig("plots/acrobot_contact_IPOPT.pdf")
	
	q_sol = state_to_configuration(x_sol)
	visualize!(vis, acrobot_impact, q_sol, Δt=h);
end
