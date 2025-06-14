# InteriorPointDDP.jl

This repository contains a Julia package for solving constrained optimal control problems (OCPs) with our proposed IPDDP2 algorithm. The arXiv preprint is located at https://arxiv.org/abs/2504.08278.

OCPs must be expressed in the form
```math
\begin{array}{rl}
    \underset{\mathbf{x}, \mathbf{u}}{\text{minimize}}  & J(\mathbf{x}, \mathbf{u}) = \sum_{t=1}^{N} \ell^t(x_t, u_t) \\
    \text{subject to} & x_1 = \hat{x}_1, \\
    & x_{t+1} = f^t(x_t, u_t) \quad \text{for } t \in\{1,\dots, N-1\}, \\
    & c^t(x_t, u_t) = 0 \quad \text{for } t \in\{1,\dots, N\}, \\
    & b_L \leq u_t \leq b_U \quad \text{for } t \in\{1,\dots, N\},
\end{array}
```
where $\mathbf{x} = (x_1, \dots, x_{N})$ is the state trajectory and $\mathbf{u} = (u_1, \dots, u_{N})$ is the control trajectory. Bounds are specified as $b_L \in [-\infty, \infty)$ and $b_U \in (-\infty, \infty]$.

- All derivatives, including Jacobians and Hessians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs/objectives, constraints, and dynamics.

- The size of the state and control vectors may change across timesteps.

## Quick Start

1. Change directory to the `ipddp2` directory, i.e., `cd /path/to/InteriorPointDDP.jl/experiments/ipddp2/`.

2. Open the Julia REPL with this project, i.e., `julia --project=../..`.

3. Switch to package mode using `]` and type `instantiate` if you are loading this project for the first time to download and install dependencies.

4. Run one of the experiments from the paper from the Julia REPL, e.g., `include("concar.jl")`.

See `src/options.jl` for full list of solver options which can be modified to improve the convergence performance of IPDDP2.
