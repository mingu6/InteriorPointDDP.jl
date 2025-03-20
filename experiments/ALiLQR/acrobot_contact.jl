using IterativeLQR
using LinearAlgebra
using Random
using Plots
using MeshCat
using BenchmarkTools
using Printf
using LaTeXStrings

visualise = false
benchmark = false
verbose = true

Δ = 0.05
N = 101
n_ocp = 500

qN = [π; 0.0]

options = Options()
options.max_dual_updates = 2
options.initial_constraint_penalty = 1e-6
options.scaling_penalty = 1.02
options.max_iterations = 300
options.min_step_size = 1e-8

include("../models/acrobot.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

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

	xN = [qN; qN]

	# ## Dynamics - implicit variational integrator (midpoint)

	dyn_acrobot = Dynamics((x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]], nx, nu)
	dynamics = [dyn_acrobot for k = 1:N-1]

	# ## Costs

	function stage_cost(x, u)
		τ = u[1]
		J = 0.01 * Δ * τ * τ
		return J
	end

	function term_cost(x, u)
		J = 0.0 
		
		q⁻ = x[1:acrobot_impact.nq] 
		q = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
		q̇ᵐ⁻ = (q - q⁻) ./ Δ

		J += 200.0 * dot(q̇ᵐ⁻, q̇ᵐ⁻)
		J += 500.0 * dot(q - qN, q - qN)
		return J
	end

	stage = Cost(stage_cost, nx, nu)
	objective = [[stage for k = 1:N-1]..., Cost(term_cost, nx, 0)]

	function eval_objective(x, u)
		J = 0.0
		for t = 1:N-1
			J += stage_cost(x[t], u[t])
		end
		J += term_cost(x[N], 0.0)
	end

	# ## Constraints - perturb complementarity to make easier
	
	limit = 5.0 * rand() + 10.0  # bound is in [10, 15]
	path_constr_fn = (x, u) -> [
		implicit_contact_dynamics(acrobot_impact, x, u, Δ, 0.0);
		u[1] - limit;
		-u[1] - limit;
		-u[nτ + nq + 1:end]
		]
	path_constr = Constraint(path_constr_fn, nx, nu)

	constraints = [[path_constr for k = 1:N-1]..., Constraint()]

	function eval_constraints_1norm(x, u)
		θ = 0.0
		for t in 1:N-1
			h = path_constr_fn(x[t], u[t])
			θ += norm(max.(h[7:end], zeros(6)), 1)
			θ += norm(h[1:6], 1)
		end
		return θ
	end
					
	# ## Initialise solver

	q1 = 0.1 .* (rand(2) .- 0.5)
	q1_plus = 0.1 .* (rand(2) .- 0.5)
	x1 = [q1; q1_plus]

	solver = Solver(dynamics, objective, constraints; options=options)
	solver.options.verbose = verbose

	q1 = 0.1 .* (rand(2) .- 0.5)
	q_init = LinRange(q1, qN, N)[2:end]
	ū = [[1.0e-1 * (rand(nτ) .- 0.5); q_init[k]; 0.01 * ones(nc); 0.01 * ones(nc)] for k = 1:N-1]
	x̄ = rollout(dynamics, x1, ū)
	
	solve!(solver, x̄, ū)

	x_sol, u_sol = get_trajectory(solver)

	J = eval_objective(x_sol, u_sol)
	θ = eval_constraints_1norm(x_sol, u_sol)
	
	if benchmark
		solver.options.verbose = false
		solve_time = @belapsed solve!(solver, $x̄, $ū) samples=10 setup=(solver=Solver($dynamics, $objective, $constraints; options=$options))
		push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ, solve_time * 1000, 0.0])
    else
        push!(results, [seed, solver.data.iterations[1], solver.data.status[1], J, θ])
    end
end

open("results/acrobot_contact.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
                            results[i][4], results[i][5], results[i][6] * 1000, results[i][7] * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]), results[i][4], results[i][5])
        end
    end
end
