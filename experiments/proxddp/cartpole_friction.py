import numpy as np

import aligator
from aligator import constraints, dynamics, manifolds
import casadi as cs

dt = 0.05
N = 101

nq = 2
nx = 2 * nq
nu = 15
N = 101

tol = 1e-4
mu_init = 0.05

verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
solver.max_iters = 2000
solver.reg_min = 1e-5

g = 9.81

res = []

with open("../ipddp2/params/cartpole_friction.txt", 'r') as file:
    for exper_ind, line in enumerate(file):
        params = [float(num_str) for num_str in line.split()]
        mc, mp, l, cfslide, cfarm = params

        class Constraint(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 8)
                q = cs.SX.sym('q', 2)
                qdot = cs.SX.sym('qdot', 2)

                M = cs.SX(2, 2)
                M[0, 0] = mc + mp
                M[0, 1] = l + cs.cos(q[1])
                M[1, 0] = mp * l * cs.cos(q[1])
                M[1, 1] = mp * l ** 2.0
                Mf = cs.Function('M', [q], [M])

                B = np.array([1.0, 0.0])
                P = np.eye(2)
                C = cs.SX(2, 2)
                C[0, 1] = -mp * qdot[1] * l * cs.sin(q[1])
                G = cs.vertcat(0.0, -mp * g * l * cs.sin(q[1]))
                Cf = cs.Function('C', [q, qdot], [C @ qdot - G])

                qm = cs.SX.sym('qm', 2)
                qc = cs.SX.sym('qc', 2)
                x = cs.vertcat(qm, qc)

                qp = cs.SX.sym('qp', 2)
                qdp = (qp - qc) / dt
                qdm = (qc - qm) / dt
                qmm = 0.5 * (qm + qc)
                qmp = 0.5 * (qp + qc)

                F = cs.SX.sym('F', 1)
                beta1 = cs.SX.sym('beta1', 2)
                beta2 = cs.SX.sym('beta2', 2)
                psi = cs.SX.sym('psi', 2)
                sc = cs.SX.sym('sc', 6)
                u = cs.vertcat(F, qp, beta1, beta2, psi, sc)

                lam = cs.vertcat(beta1[0] - beta1[1], beta2[0] - beta2[1])
                gamma1 = cfslide * (mp + mc) * g * dt
                gamma2 = cfarm * mp * g * l * dt

                Mhat = Mf(qmp) @ qdp - Mf(qmm) @ qdm
                Chat = 0.5 * (Cf(qmp, qdp) + Cf(qmm, qdm))
                dyn = Mhat + dt * (Chat - B * F - P.T @ lam)

                eta1 = cs.vertcat(
                    qdp[0] + psi[0],
                    -qdp[0] + psi[0]
                )
                eta2 = cs.vertcat(
                    qdp[1] + psi[1],
                    -qdp[1] + psi[1]
                )

                constr = cs.vertcat(
                    dyn,
                    psi[0] * (gamma1 - cs.sum(beta1)) - sc[0],
                    psi[1] * (gamma2 - cs.sum(beta2)) - sc[1],
                    beta1 * eta1 - sc[2:4],
                    beta2 * eta2 - sc[4:6]
                )
                Jx = cs.jacobian(constr, x)
                Ju = cs.jacobian(constr, u)
                self.constr = cs.Function('constr', [x, u], [constr])
                self.Jx = cs.Function('Jx', [x, u], [Jx])
                self.Ju = cs.Function('Ju', [x, u], [Ju])

            def evaluate(self, x, u, data):
                data.value[:] = self.constr(x, u).toarray()[:, 0]

            def computeJacobians(self, x, u, data):
                data.Jx[:, :] = self.Jx(x, u).toarray()
                data.Ju[:] = self.Ju(x, u).toarray()


        class NonNeg(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 16)
                qm = cs.SX.sym('qm', 2)
                qc = cs.SX.sym('qc', 2)
                qp = cs.SX.sym('qp', 2)
                x = cs.vertcat(qm, qc)
                qdp = (qp - qc) / dt

                F = cs.SX.sym('F', 1)
                beta1 = cs.SX.sym('beta1', 2)
                beta2 = cs.SX.sym('beta2', 2)

                psi = cs.SX.sym('psi', 2)
                sc = cs.SX.sym('sc', 6)
                u = cs.vertcat(F, qp, beta1, beta2, psi, sc)

                gamma1 = cfslide * (mp + mc) * g * dt
                gamma2 = cfarm * mp * g * l * dt

                eta1 = cs.vertcat(
                    qdp[0] + psi[0],
                    -qdp[0] + psi[0]
                )
                eta2 = cs.vertcat(
                    qdp[1] + psi[1],
                    -qdp[1] + psi[1]
                )

                nonneg = cs.vertcat(
                    -(gamma1 - cs.sum(beta1)),
                    -(gamma2 - cs.sum(beta2)),
                    -eta1,
                    -eta2,
                    -beta1,
                    -beta2,
                    -sc
                )
                self.nonneg = cs.Function('nonneg', [x, u], [nonneg])
                self.Ju = cs.Function('Ju', [x, u], [cs.jacobian(nonneg, u)])
                self.Jx = cs.Function('Jx', [x, u], [cs.jacobian(nonneg, x)])

            def evaluate(self, x, u, data):
                data.value[:] = self.nonneg(x, u).toarray()[:, 0]

            def computeJacobians(self, x, u, data):
                data.Jx[:, :] = self.Jx(x, u).toarray()
                data.Ju[:, :] = self.Ju(x, u).toarray()


        class Limit(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 1)

            def evaluate(self, x, u, data):
                data.value[:] = u[0]

            def computeJacobians(self, x, u, data):
                data.Ju[:] = 0.0
                data.Jx[:] = 0.0
                data.Ju[0] = 1.0


        class Cost(aligator.CostAbstract):
            def __init__(self):
                space = manifolds.VectorSpace(nx)
                super().__init__(space, nu)
                qc = cs.SX.sym('qc', 2)
                qp = cs.SX.sym('qp', 2)

                F = cs.SX.sym('F', 1)
                beta1 = cs.SX.sym('beta1', 2)
                beta2 = cs.SX.sym('beta2', 2)

                psi = cs.SX.sym('psi', 2)
                sc = cs.SX.sym('sc', 6)
                u = cs.vertcat(F, qp, beta1, beta2, psi, sc)

                J = dt * 1e-2 * F ** 2 + 2.0 * cs.sum(sc)
                self.J = cs.Function('J', [u], [J])
                self.Ju = cs.Function('Ju', [u], [cs.jacobian(J, u)])
                self.Juu = cs.Function('Juu', [u], [cs.hessian(J, u)[0]])

            def evaluate(self, x, u, data):
                data.value = self.J(u).toarray()[0][0]

            def computeGradients(self, x, u, data):
                data.Lx[:] = 0.0
                data.Lu[:] = self.Ju(u).toarray()

            def computeHessians(self, x, u, data):
                data.Lxx[:, :] = 0.0
                data.Lxu[:, :] = 0.0
                data.Luu[:, :] = self.Juu(u).toarray()


        class TermCost(aligator.CostAbstract):
            def __init__(self):
                space = manifolds.VectorSpace(nx)
                super().__init__(space, nu)
                qm = cs.SX.sym('qm', 2)
                qc = cs.SX.sym('qc', 2)
                qdm = (qc - qm) / dt
                x = cs.vertcat(qm, qc)

                J = 200.0 * cs.sumsqr(qdm) + 700.0 * cs.sumsqr(qc - qN)
                self.obj = cs.Function('L', [x], [J])
                Lx = cs.jacobian(J, x)
                self.Lx = cs.Function('Lx', [x], [Lx])
                Lxx = cs.hessian(J, x)
                self.Lxx = cs.Function('Lxx', [x], Lxx)

            def evaluate(self, x, u, data):
                data.value = self.obj(x).toarray()[0][0]

            def computeGradients(self, x, u, data):
                data.Lu[:] = 0.0
                data.Lx[:] = self.Lx(x).toarray()

            def computeHessians(self, x, u, data):
                data.Lxx[:, :] = self.Lxx(x)[0].toarray()
                data.Lxu[:, :] = 0.0
                data.Luu[:, :] = 0.0

        x0 = np.zeros(nx)
        qN = np.array([0.0, np.pi])
        space = manifolds.VectorSpace(nx)

        term_cost = TermCost()
        problem = aligator.TrajOptProblem(x0, nu, space, term_cost=term_cost)

        A = np.zeros((nx, nx))
        A[:2, 2:] = np.eye(2)
        B = np.zeros((nx, nu))
        B[2:, 1:3] = np.eye(2)
        dynmodel = dynamics.LinearDiscreteDynamics(A, B, np.zeros(nx))

        rcost = Cost()
        constr = Constraint()
        nonneg = NonNeg()
        limit = Limit()

        for i in range(N-1):
            stage = aligator.StageModel(rcost, dynmodel)
            stage.addConstraint(constr, constraints.EqualityConstraintSet())
            stage.addConstraint(nonneg, constraints.NegativeOrthant())
            stage.addConstraint(limit, constraints.BoxConstraint(
                np.array([-10.0]),
                np.array([10.0]))
                )
            problem.addStage(stage)

        us_init = [np.concatenate((np.zeros(3), 0.01 * np.ones(12)))] * (N-1)
        xs_init = aligator.rollout(dynmodel, x0, us_init)

        solver.setup(problem)
        solver.run(problem, xs_init, us_init)

        results = solver.results

        converged = 'true' if results.conv else 'false'
        res.append([str(exper_ind+1), str(results.num_iters), converged,
                    str(results.traj_cost), str(results.primal_infeas)])

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

with open("results/cartpole_friction.txt", 'w') as file:
    for line in res:
        file.write(f"{' '.join(line)}\n")

# 0.2
# Average number of iterations:  400.0
# Average cost:  -26.579939469264307
# Average violation:  1.268376354684198
# Successes:  0

# 0.1
# Average number of iterations:  400.0
# Average cost:  -15.466324732092522
# Average violation:  0.9680251436890063
# Successes:  0

# 0.05
# Average number of iterations:  400.0
# Average cost:  -11.174169111939454
# Average violation:  0.1969890013851635
# Successes:  0

# 0.02
# Average number of iterations:  400.0
# Average cost:  -5.173604400506558
# Average violation:  0.07509319788098151
# Successes:  0