import numpy as np

import aligator
from aligator import constraints, dynamics, manifolds
import casadi as cs

dt = 0.01

class Constraint(aligator.StageFunction):
    def __init__(self) -> None:
        super().__init__(nx, nu, 1)
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        c = u[1] - u[2] - u[0] * x[1]
        Jx = cs.jacobian(c, x)
        Ju = cs.jacobian(c, u)
        self.c = cs.Function('c', [x, u], [c])
        self.Jx = cs.Function('Jx', [x, u], [Jx])
        self.Ju = cs.Function('Ju', [x, u], [Ju])

    def evaluate(self, x, u, data):
        data.value[:] = self.c(x, u).toarray()[0]

    def computeJacobians(self, x, u, data):
        data.Jx[:] = self.Jx(x, u).toarray()
        data.Ju[:] = self.Ju(x, u).toarray()


class NonNeg(aligator.StageFunction):
    def __init__(self) -> None:
        super().__init__(nx, nu, 2)

    def evaluate(self, x, u, data):
        data.value[:] = -u[1:]

    def computeJacobians(self, x, u, data):
        data.Jx[:, :] = 0.0
        data.Ju[:, :] = 0.0
        data.Ju[0, 1] = -1.0
        data.Ju[1, 2] = -1.0

    def computeVectorHessianProducts(self, x, u, lbda, data):
        data.Hxx[:, :] = 0.0
        data.Huu[:, :] = 0.0
        data.Hxu[:, :] = 0.0


class Limit(aligator.StageFunction):
    def __init__(self) -> None:
        super().__init__(nx, nu, 1)

    def evaluate(self, x, u, data):
        data.value[:] = u[0]

    def computeJacobians(self, x, u, data):
        data.Jx[:] = 0.0
        data.Ju[:] = 0.0
        data.Ju[0] = 1.0


class Cost(aligator.CostAbstract):
    def __init__(self):
        space = manifolds.VectorSpace(nx)
        super().__init__(space, nu)

    def evaluate(self, x, u, data):
        data.value = dt * (u[1] + u[2])

    def computeGradients(self, x, u, data):
        data.Lx[:] = 0.0
        data.Lu[0] = 0.0
        data.Lu[1] = dt
        data.Lu[2] = dt

    def computeHessians(self, x, u, data):
        data.Lxx[:, :] = 0.0
        data.Lxu[:, :] = 0.0
        data.Luu[:, :] = 0.0


class Dynamics(dynamics.ODEAbstract):
    def __init__(self) -> None:
        space = manifolds.VectorSpace(nx)
        super().__init__(space, nu)

    def forward(self, x, u, data):
        data.xdot[0] = x[1]
        data.xdot[1] = u[0]

    def dForward(self, x, u, data):
        data.Ju[:, :] = 0.0
        data.Ju[1, 0] = 1.0
        data.Jx[:, :] = 0.0
        data.Jx[0, 1] = 1.0

nx = 2
nu = 3
N = 101

tol = 1e-5
mu_init = 0.01

x0 = np.zeros(2)
x_tar = np.array([1.0, 0.0])
wterm = np.array([500.0, 500.0])
space = manifolds.VectorSpace(nx)

term_cost = aligator.QuadraticStateCost(space, nu, x_tar, 2 * np.diag(wterm))
problem = aligator.TrajOptProblem(x0, nu, space, term_cost=term_cost)
dynmodelc = Dynamics()
dynmodel = dynamics.IntegratorEuler(dynmodelc, dt)

# bound constraint

rcost = Cost()
constr = Constraint()
nonneg = NonNeg()
limit = Limit()

for i in range(N-1):
    stage = aligator.StageModel(rcost, dynmodel)
    stage.addConstraint(constr, constraints.EqualityConstraintSet())
    stage.addConstraint(nonneg, constraints.NegativeOrthant())
    stage.addConstraint(limit, constraints.BoxConstraint(np.array([-10.0]), np.array([10.0])))
    problem.addStage(stage)

us_init = [np.array([0.01, 0.01, 0.01])] * (N-1)
xs_init = aligator.rollout(dynmodel, x0, us_init)

verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
history_cb = aligator.HistoryCallback(solver)
solver.max_iters = 2000
solver.reg_min = 1e-5
solver.registerCallback("his", history_cb)

solver.setup(problem)
solver.run(problem, xs_init, us_init)

results = solver.results
print(results)

# Convergence failure.
# Results {
#   num_iters:    347,
#   converged:    false,
#   traj. cost:   1.266e+00,
#   merit.value:  1.266e+00,
#   prim_infeas:  8.899e-11,
#   dual_infeas:  7.979e-08,
#   al_iters:     100,
# }
