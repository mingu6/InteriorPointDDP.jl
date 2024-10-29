using InteriorPointDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using BenchmarkTools
using Printf

visualise = false
benchmark = true
verbose = true

T = Float64
h = 0.05
N = 101

include("models/cartpole.jl")

if visualise
	include("visualise/visualise_cartpole.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

nq = cartpole.nq
nu = cartpole.nu
nx = 2 * nq
ny = nu + nq  # torque and acceleration now decision variables/"controls"

xN = T[0.0; π; 0.0; 0.0]

options = Options{T}(quasi_newton=false, verbose=true)

# ## Dynamics - implicit dynamics with RK2 integration

f = (x, y) -> [x[nq .+ (1:nq)]; y[nu .+ (1:nq)]]
cartpole_discrete = (x, y) -> x + h * f(x + 0.5 * h * f(x, y), y)  # Explicit midpoint
cartpole_dyn = Dynamics(cartpole_discrete, nx, ny)
dynamics = [cartpole_dyn for k = 1:N-1]

# ## Costs

stage = Cost((x, y) -> h * dot(y[1], y[1]), nx, ny)
objective = [
    [stage for k = 1:N-1]...,
    Cost((x, y) -> 400.0 * dot(x - xN, x - xN), nx, 0)
] 

# ## Constraints

stage_constr = Constraint((x, y) -> implicit_dynamics(cartpole, x, y) * h, nx, ny)

constraints = [stage_constr for k = 1:N-1]

# ## Bounds

bound = Bound(
	[-T(4.0) * ones(T, nu); -T(Inf) * ones(T, nq)],
	[T(4.0) * ones(T, nu); T(Inf) * ones(T, nq)]
)
bounds = [bound for k in 1:N-1]

solver = Solver(T, dynamics, objective, constraints, bounds, options=options)

open("examples/results/cartpole_implicit.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        time (s)  \n")
    for seed = 1:50
        solver.options.verbose = verbose
        Random.seed!(seed)
        
        # ## Initialise solver and solve
        
        x1 = T[0.0; 0.0; 0.0; 0.0] + (rand(T, 4) .- T(0.5)) .* T[0.05, 0.2, 0.1, 0.1]
        ū = [T(1.0e-2) * (rand(T, ny) .- T(0.5)) for k = 1:N-1]
        solve!(solver, x1, ū)
        
        if benchmark
            solver.options.verbose = false
            solve_time = @belapsed solve!($solver, $x1, $ū)
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e    %.5f  \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf, solve_time)
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
