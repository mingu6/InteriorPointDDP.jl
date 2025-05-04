import numpy as np

import aligator
from aligator import constraints, dynamics, manifolds
import casadi as cs

nx = 4
nu = 6
N = 101

dt = 0.05
r_car = 0.02

tol = 1e-4
mu_init = 0.1

verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
solver.max_iters = 2000
solver.reg_min = 1e-5

res = []

with open("../ipddp2/params/concar.txt", 'r') as file:
    for exper_ind, line in enumerate(file):
        params = [float(num_str) for num_str in line.split()]
        F_lim, tau_lim = params[:2]
        obs_1 = params[2:5]
        obs_2 = params[5:8]
        obs_3 = params[8:11]
        obs_4 = params[11:14]
        x0 = np.array(params[14:18])

        class Dynamics(dynamics.ODEAbstract):
            def __init__(self) -> None:
                space = manifolds.VectorSpace(nx)
                super().__init__(space, nu)
                x = cs.SX.sym('x', nx)
                F = cs.SX.sym('F', 1)
                tau = cs.SX.sym('tau', 1)
                s = cs.SX.sym('s', 4)
                u = cs.vertcat(F, tau, s)

                dyn = cs.vertcat(x[3] * cs.cos(x[2]), x[3] * cs.sin(x[2]), tau, F)
                self.dyn = cs.Function('dyn', [x, u], [dyn])
                Jx = cs.jacobian(dyn, x)
                Ju = cs.jacobian(dyn, u)
                self.Jx = cs.Function('Jx', [x, u], [Jx])
                self.Ju = cs.Function('Ju', [x, u], [Ju])

            def forward(self, x, u, data):
                data.xdot[:] = self.dyn(x, u).toarray()[:, 0]

            def dForward(self, x, u, data):
                data.Jx[:, :] = self.Jx(x, u).toarray()
                data.Ju[:, :] = self.Ju(x, u).toarray()


        class Limit(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 2)

            def evaluate(self, x, u, data):
                data.value[:] = u[:2]

            def computeJacobians(self, x, u, data):
                data.Ju[:] = 0.0
                data.Jx[:] = 0.0
                data.Ju[0, 0] = 1.0
                data.Ju[1, 1] = 1.0


        class NonNeg(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 8)
                x = cs.SX.sym('x', nx)
                F = cs.SX.sym('F', 1)
                tau = cs.SX.sym('tau', 1)
                s = cs.SX.sym('s', 4)
                u = cs.vertcat(F, tau, s)

                obs = [
                    np.array(obs_1),
                    np.array(obs_2),
                    np.array(obs_3),
                    np.array(obs_4)
                ]

                obs_constr = cs.vertcat(*[(obs[i][2] + r_car) ** 2 - cs.sumsqr(x[:2] - obs[i][:2]) - s[i] for i in range(4)])
                constr = cs.vertcat(obs_constr, -s)
                self.constr = cs.Function('constr', [x, u], [constr])
                self.Jx = cs.Function('Jx', [x, u], [cs.jacobian(constr, x)])
                self.Ju = cs.Function('Ju', [x, u], [cs.jacobian(constr, u)])

            def evaluate(self, x, u, data):
                data.value[:] = self.constr(x, u).toarray()[:, 0]

            def computeJacobians(self, x, u, data):
                data.Jx[:, :] = self.Jx(x, u).toarray()
                data.Ju[:, :] = self.Ju(x, u).toarray()


        class Cost(aligator.CostAbstract):
            def __init__(self):
                space = manifolds.VectorSpace(nx)
                super().__init__(space, nu)
                F = cs.SX.sym('F', 1)
                tau = cs.SX.sym('tau', 1)
                s = cs.SX.sym('s', 4)
                u = cs.vertcat(F, tau, s)

                L = dt * (5.0 * F ** 2 + tau ** 2) + 1000. * cs.sumsqr(s)
                self.L = cs.Function('L', [u], [L])
                self.Lu = cs.Function('Lu', [u], [cs.jacobian(L, u)])
                self.Luu = cs.Function('Luu', [u], [cs.hessian(L, u)[0]])

            def evaluate(self, x, u, data):
                data.value = self.L(u).toarray()[0][0]

            def computeGradients(self, x, u, data):
                data.Lx[:] = 0.0
                data.Lu[:] = self.Lu(u).toarray()

            def computeHessians(self, x, u, data):
                data.Lxx[:, :] = 0.0
                data.Lxu[:, :] = 0.0
                data.Luu[:, :] = self.Luu(u)

                
        x_tar = np.array([1.0, 1.0, np.pi / 4, 0.0])
        wterm = 200.0 * np.ones(4)
        space = manifolds.VectorSpace(nx)

        stage_cost = Cost()
        term_cost = aligator.QuadraticStateCost(space, nu, x_tar, 2 * np.diag(wterm))
        dynmodelc = Dynamics()
        dynmodel = dynamics.IntegratorRK2(dynmodelc, dt)

        nonneg = NonNeg()
        limit = Limit()

        problem = aligator.TrajOptProblem(x0, nu, space, term_cost=term_cost)

        for i in range(N-1):
            stage = aligator.StageModel(stage_cost, dynmodel)
            stage.addConstraint(nonneg, constraints.NegativeOrthant())
            stage.addConstraint(limit, constraints.BoxConstraint(
                np.array([-F_lim, -tau_lim]),
                np.array([F_lim, tau_lim]))
                )
            problem.addStage(stage)

        us_init = [np.array([0., 0., 0.01, 0.01, 0.01, 0.01])] * (N-1)
        xs_init = aligator.rollout(dynmodel, x0, us_init)

        solver.setup(problem)
        solver.run(problem, xs_init, us_init)

        results = solver.results

        converged = 'true' if results.conv else 'false'
        res.append([str(exper_ind+1), str(results.num_iters), converged,
                    str(results.traj_cost), str(results.primal_infeas)])


with open("results/concar_quad.txt", 'w') as file:
    for line in res:
        file.write(f"{' '.join(line)}\n")


import matplotlib.pyplot as plt
x_res = results.xs
xs_ = []
ys_ = []
for x in x_res:
    x_, y_ = x[:2]
    xs_.append(x_)
    ys_.append(y_)

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.plot(xs_, ys_)
circ1 = plt.Circle(obs_1[:2], obs_1[2])
circ2 = plt.Circle(obs_2[:2], obs_2[2])
circ3 = plt.Circle(obs_3[:2], obs_3[2])
circ4 = plt.Circle(obs_4[:2], obs_4[2])
ax.add_patch(circ1)
ax.add_patch(circ2)
ax.add_patch(circ3)
ax.add_patch(circ4)
plt.savefig("concar_test.png")

iters = 0
cost = 0.0
viol = 0.0
succ = 0
for r in res:
    iters += int(r[1])
    cost += float(r[3])
    viol += float(r[4])
    if r[2] == 'true':
        succ += 1

print("Average number of iterations: ", iters / len(res))
print("Average cost: ", cost / len(res))
print("Average violation: ", viol / len(res))
print("Successes: ", succ)
