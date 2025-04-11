# InteriorPointDDP.jl

This repository contains a Julia package for solving constrained optimal control problems (OCPs) with our proposed IPDDP2 algorithm.

OCPs must be expressed in the form
```math
\begin{array}{rl}
    \underset{\mathbf{x}, \mathbf{u}}{\text{minimize}}  & J(\mathbf{x}, \mathbf{u}) \coloneqq \sum_{t=0}^{N-1} \ell(x_t, u_t) + \ell^F(x_N) \\
    \text{subject to} & x_0 = \bar{x}_0, \\
    & x_{t+1} = f(x_t, u_t), \\
    & h(x_t, u_t) = 0, \\
    & b_L \leq u_t \leq b_U \quad \text{for } t \in\{0,\dots, N-1\},
\end{array}
```
where $\mathbf{x} \coloneqq (x_0, \dots, x_{N})$ is the state trajectory and $\mathbf{u} \coloneqq (u_0, \dots, u_{N-1})$ is the control trajectory. Bounds are specified as $b_L \in [-\infty, \infty)$ and $b_U \in (-\infty, \infty]$.

- All derivatives, including Jacobians and Hessians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs/objectives, constraints, and dynamics.

- The size of the state and control vectors may change across timesteps. Constraints, costs and dynamics can be time-varying.

## Quick Start

1. Change directory to the `ipddp2` directory, i.e., `cd /path/to/InteriorPointDDP.jl/experiments/ipddp2/`.

2. Open the Julia REPL with this project, i.e., `julia --project=../..`.

3. Switch to package mode using `]` and type `instantiate` if you are loading this project for the first time to download and install dependencies.

4. Run one of the experiments from the paper from the Julia REPL, e.g., `include("concar.jl")`.

See `src/options.jl` for full list of solver options which can be modified to improve the convergence performance of IPDDP2.
