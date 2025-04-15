using JuMP
import Ipopt
using Random
using Plots
using MeshCat
using Suppressor
using Printf
using LaTeXStrings

visualise = false
output = false
benchmark = false
n_benchmark = 10

print_level = output ? 5 : 4

Δ = 0.05
N = 101
n_ocp = 500

include("../models/acrobot.jl")
include("ipopt_parse.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

qN = [π; 0.0]
xN = [qN; qN]

n_ocp = 500

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
	Random.seed!(seed)

	acrobot_impact = DoublePendulum{Float64}(2, 1, 2,
		0.9 + 0.2 * rand(),
		0.3 + 0.1 * rand(), 
		0.9 + 0.2 * rand(),
		0.4 + 0.2 * rand(),
		0.9 + 0.2 * rand(),
		0.3 + 0.1 * rand(), 
		0.9 + 0.2 * rand(),
		0.4 + 0.2 * rand(),
		9.81, 0.0, 0.0)

	nq = acrobot_impact.nq
	nc = acrobot_impact.nc
	nτ = acrobot_impact.nu
	nx = 2 * nq
	nu = nτ + nq + 2 * nc

	model = Model(
		optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6,
			"print_level" => print_level, "print_timing_statistics" => "yes")
		);

	f = (x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]]

	# ## Objective

	function stage_obj(x, u)
		τ = u[1]
		J = 0.01 * Δ * τ * τ
		return J
	end

	function term_obj(x, u)
		J = 0.0 
		
		q⁻ = x[1:acrobot_impact.nq] 
		q = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
		q̇ᵐ⁻ = (q - q⁻) ./ Δ

		J += 200.0 * q̇ᵐ⁻' * q̇ᵐ⁻
		J += 500.0 * (q - qN)' * (q - qN)
		return J
	end

	cost = (x, u) -> begin
		J = 0.0
		for k in 1:N-1
			J += stage_obj(x[k, :], u[k, :])
		end
		J += term_obj(x[N, :], 0.0)
	end

	# ## Constraints

	constr = (x, u) -> implicit_contact_dynamics(acrobot_impact, x, u, Δ, 1e-7)

	@variable(model, x[1:N, 1:nx]);
	@variable(model, u[1:N-1, 1:nu]);

	limit = 5.0 * rand() + 10.0  # bound is in [10, 15]
	for t = 1:N-1
		set_lower_bound(u[t, 1], -limit)
		set_upper_bound(u[t, 1], limit)
		for i = (nτ + nq) .+ (1:2*nc)
			set_lower_bound(u[t, i], 0.0)
		end
	end

	@objective(model, Min, cost(x, u))


	q1 = 0.1 .* (rand(2) .- 0.5)
	q1_plus = 0.1 .* (rand(2) .- 0.5)
	x1 = [q1; q1_plus]
	fix.(x[1, :], x1, force = true)
	for k = 1:N-1
		@constraint(model, x[k+1, :] == f(x[k, :], u[k, :]))
		@constraint(model, constr(x[k, :], u[k, :]) .== 0.0)
	end
	
	# ## Initialise variables and solve
	
	q1 = 0.1 .* (rand(2) .- 0.5)
	q_init = LinRange(q1, qN, N)[2:end]
	ū = [[1.0e-1 * (rand(nτ) .- 0.5); q_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
	
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
	
	set_attribute(model, "print_level", print_level)
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

	# ## Plot solution
	if seed == 1
		xv = value.(x)
		x_sol = [xv[k, :] for k in 1:N]
		uv = value.(u)
		u_sol = [uv[k, :] for k in 1:N-1]

		θe = map(x -> x[4], x_sol[1:end-1])
		s1 = map(θ -> π / 2 - θ, θe)
		s2 = map(θ -> θ + π / 2, θe)
		λ1 = map(u -> u[4], u_sol)
		λ2 = map(u -> u[5], u_sol)
		plot(range(0, Δ * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylims=(0,6),
		legendfontsize=12, linewidth=2, xlabelfontsize=14, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], 
		background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
		savefig("plots/acrobot_contact_IPOPT.pdf")
	end

	# ## Visualise trajectory using MeshCat

	if visualise && seed == 1
		q_sol = state_to_configuration(x_sol)
		visualize!(vis, acrobot_impact, q_sol, Δt=Δ);
	end
end

open("results/acrobot_contact.txt", "w") do io
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

