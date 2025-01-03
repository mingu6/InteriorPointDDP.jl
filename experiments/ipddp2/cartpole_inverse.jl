using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf

visualise = false
benchmark = false
verbose = true
quasi_newton = false
n_benchmark = 10

T = Float64
h = 0.05
N = 101

include("../models/cartpole.jl")

if visualise
	include("../visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = cartpole.nq
nF = cartpole.nu
nx = 2 * nq
nu = nF + nq  # torque and acceleration now decision variables/"controls"

xN = T[0.0; π; 0.0; 0.0]

options = Options{T}(quasi_newton=quasi_newton, verbose=false)

# ## Dynamics - forward Euler

f = (x, u) -> x + h * [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]
cartpole_dyn = Dynamics(f, nx, nu)
dynamics = [cartpole_dyn for k = 1:N-1]

# ## Objective

stage = Objective((x, u) -> 0.1 * h * dot(u[1], u[1]), nx, nu)
objective = [
    [stage for k = 1:N-1]...,
    Objective((x, u) -> 100.0 * dot(x - xN, x - xN), nx, 0)
] 

# ## Constraints

stage_constr = Constraint((x, u) -> implicit_dynamics(cartpole, x, u) * h, nx, nu)

constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(
	[-T(4.0) * ones(T, nF); -T(Inf) * ones(T, nq)],
	[T(4.0) * ones(T, nF); T(Inf) * ones(T, nq)]
)
bounds = [bound for k in 1:N-1]

solver = Solver(T, dynamics, objective, constraints, bounds, options=options)

fname = quasi_newton ? "results/cartpole_inverse_QN.txt" : "results/cartpole_inverse.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for seed = 1:50
        solver.options.verbose = verbose
        Random.seed!(seed)
        
        # ## Initialise solver and solve
        
        x1 = (rand(T, 4) .- T(0.5)) .* T[0.05, 0.05, 0.05, 0.05]
        ū = [T(1.0e-1) * (rand(T, nu) .- T(0.5)) for k = 1:N-1]
        solve!(solver, x1, ū)

        if benchmark
            solver.options.verbose = false
            solver_time = 0.0
            wall_time = 0.0
            for i in 1:n_benchmark
                solve!(solver, x1, ū)
                solver_time += solver.data.solver_time
                wall_time += solver.data.wall_time
            end
            solver_time /= n_benchmark
            wall_time /= n_benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", seed, solver.data.k, solver.data.status == 0,
                    solver.data.objective, solver.data.primal_inf, wall_time * 1000, solver_time * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        end
    end
end

# ## Visualise solution

if visualise
    x_sol, u_sol = get_trajectory(solver)
    
	q_sol = [x[1:nq] for x in x_sol]
	visualize!(vis, cartpole, q_sol, Δt=h);
end
