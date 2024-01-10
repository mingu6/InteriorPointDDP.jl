# InteriorPointDDP.jl (Work in progress!!!)

Implemented by Ming Xu, Pranav Pativada, Jeffrey Liang and Iman Shames.

A Julia package for solving constrained trajectory optimization problems with interior point DDP (IPDDP) [[Paper](https://arxiv.org/pdf/2004.12710), [Original Code](https://github.com/Xapavlov/ipddp)]. We leverage the nice solver API and code structure from [IterativeLQR.jl](https://github.com/thowell/IterativeLQR.jl) for InteriorPointDDP.jl.

Problems that we can solve using IPDDP must be formulated as

$$ 
\begin{align*}
		\underset{x_{1:T}, \phantom{\,} u_{1:T-1}}{\text{minimize }} & \phantom{,} g_T(x_T; \theta_T) + \sum \limits_{t = 1}^{T-1} g_t(x_t, u_t; \theta_t)\\
		\text{subject to } & \phantom{,} f_t(x_t, u_t; \theta_t) = x_{t+1}, \phantom{,} \quad t = 1,\dots,T-1,\\
		& \phantom{,} c_t(x_t, u_t; \theta_t) \phantom{,}\leq \phantom{,} 0, \quad t = 1, \dots, T-1,\\
\end{align*}
$$

with

- $x_{1:T}$: state trajectory 
- $u_{1:T-1}$: action trajectory 
- $\theta_{1:T}$: problem-data trajectory 


- Jacobians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs, constraints, and dynamics. 

- Constraints are handled using an interior point framework.

- Cost, dynamics, and constraints can have varying dimensions at each time step.

- TODO: Allocation-free jacobians. Currently, these are causing gnarly bugs so we have removed them. 

- TODO: Implement equality constraints and terminal constraints (including inequality).

## Objectives of this project/repo

### Cleaner API (vs the MATLAB implementation)

We are closely following the code structure of IterativeLQR.jl. Benefits include
- More user friendly (tidier API)
- Easier to directly (and fairly) compare IPDDP and penalty-based methods for handling constraints (e.g., [Altro.jl](https://github.com/RoboticExplorationLab/Altro.jl), [IterativeLQR.jl](https://github.com/thowell/IterativeLQR.jl)). 
- Ease of integration into awesome contact dynamics work from the robot exploration lab (e.g., [optimization_dynamics](https://github.com/thowell/optimization_dynamics), [Dojo](https://github.com/dojo-sim/Dojo.jl)).

### Improved computation times

Implementing IPDDP in Julia will dramatically speed up computation time compared to the original MATLAB code (after compilation).

### Solver enhancements (à la IPOPT)

The original implementation in the MATLAB code uses a very basic line search filter. We believe this line search filter may not be sufficiently robust for problems which are significantly more difficult than the examples presented in the original paper (e.g., non-linear constraints). 

We propose to implement enhancements similar to state-of-the-art, general, interior point method IPOPT (see [paper](https://link.springer.com/article/10.1007/s10107-004-0559-y) for details) to improve solver robustness and convergence speed. Key enhancements include

- Second-order corrections for descent directions
- Improved regularization of descent directions
- Feasibility restoration phase
- and more....

So far, **none have been implemented**. The current implementation is similar (though not identical) to the original MATLAB implementation.

**If you have a tricky constrained trajectory optimisation problem you would like us to try, please raise an issue with the mathematical formulation/code!**

## Installation
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add https://github.com/mingu6/InteriorPointDDP.jl
```

## Quick Start

Load the Julia REPL and this project.

`julia --project=/path/to/InteriorPointDDP.jl`

Switch to package mode using `]` and type `instantiate` if you are loading this project for the first time to download and install dependencies.

Run example from the Julia REPL, e.g.,

`include("examples/ipddp/car.jl")`

Change solver options in `src/options.jl` (e.g., feasible/infeasible start, number of iterations etc.). Note you will need to recompile the project after changing this (TODO: fix this).
