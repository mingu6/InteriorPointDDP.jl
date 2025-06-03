import numpy as np

import aligator
from aligator import constraints, dynamics, manifolds
import casadi as cs

N = 101
dt = 0.05
g = 9.81

nq = 2
nc = 2
nx = 2 * nq
nu = 1 + nq + 2 * nc

tol = 1e-5
mu_init = 0.03

verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
solver.max_iters = 3000
solver.reg_min = 1e-5

res = []

with open("../ipddp2/params/acrobot_contact.txt", 'r') as file:
    for exper_ind, line in enumerate(file):
        params = [float(num_str) for num_str in line.split()]
        m1, I1, l1, lc1, m2, I2, l2, lc2 = params

        class Constraint(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 4)
                q = cs.SX.sym('q', 2)
                qdot = cs.SX.sym('qd', 2)

                M = cs.SX(2, 2)
                M[0, 0] = I1 + I2  + m2 * l1 ** 2 + 2.0 * m2 * l1 * lc2 * cs.cos(q[1])
                M[0, 1] = I2 + m2 * l1 * lc2 * cs.cos(q[1])
                M[1, 0] = M[0, 1]
                M[1, 1] = I2
                Mf = cs.Function('M', [q], [M])

                tau_g = cs.vertcat(
                    -m1 * g * lc1 * cs.sin(q[0]) - m2 * g * (l1 * cs.sin(q[0]) + lc2 * cs.sin(q[0] + q[1])),
                    -m2 * g * lc2 * cs.sin(q[0] + q[1])
                )

                C = cs.SX(2, 2)
                C[0, 0] = -2.0 * m2 * l1 * lc2 * cs.sin(q[1]) * qdot[1]
                C[0, 1] = -1.0 * m2 * l1 * lc2 * cs.sin(q[1]) * qdot[1]
                C[1, 0] = C[0, 1]

                B = np.array([0.0, 1.0])
                P = np.array([[0.0, -1.0], [0.0, 1.0]])
                Ct = C @ qdot - tau_g
                Ctf = cs.Function('Ct', [q, qdot], [Ct])

                qm = cs.SX.sym('qm', 2)
                qc = cs.SX.sym('qc', 2)
                x = cs.vertcat(qm, qc)

                qp = cs.SX.sym('qp', 2)
                qdp = (qp - qc) / dt
                qdm = (qc - qm) / dt
                qmm = 0.5 * (qm + qc)
                qmp = 0.5 * (qp + qc)

                tau = cs.SX.sym('tau', 1)
                lam = cs.SX.sym('lam', 2)
                sc = cs.SX.sym('sc', 2)

                u = cs.vertcat(tau, qp, lam, sc)

                phi = cs.vertcat(
                    0.5 * np.pi - qp[1], qp[1] + 0.5 * np.pi
                )
                Mhat = Mf(qmp) @ qdp - Mf(qmm) @ qdm
                Chat = 0.5 * (Ctf(qmp, qdp) + Ctf(qmm, qdm))
                dyn = Mhat + dt * (Chat - B * tau - P.T @ lam + 0.5 * qdp)

                constr = cs.vertcat(
                    dyn,
                    lam * phi - sc
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
                data.Ju[:, :] = self.Ju(x, u).toarray()


        class NonNeg(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 6)
                qp = cs.SX.sym('qp', 2)

                tau = cs.SX.sym('tau', 1)
                lam = cs.SX.sym('lam', 2)
                sc = cs.SX.sym('sc', 2)

                u = cs.vertcat(tau, qp, lam, sc)

                phi = cs.vertcat(
                    0.5 * np.pi - qp[1], qp[1] + 0.5 * np.pi
                )
                constr = cs.vertcat(
                    -phi,
                    -lam,
                    -sc
                )
                self.constr = cs.Function('c', [u], [constr])
                self.Ju = cs.Function('Ju', [u], [cs.jacobian(constr, u)])

            def evaluate(self, x, u, data):
                data.value[:] = self.constr(u).toarray()[:, 0]

            def computeJacobians(self, x, u, data):
                data.Jx[:, :] = 0.0
                data.Ju[:, :] = self.Ju(u).toarray()


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
                qp = cs.SX.sym('qp', 2)

                tau = cs.SX.sym('tau', 1)
                lam = cs.SX.sym('lam', 2)
                sc = cs.SX.sym('sc', 2)

                u = cs.vertcat(tau, qp, lam, sc)

                J = dt * 1e-2 * tau ** 2 + 2.0 * cs.sum(sc)
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
                np.array([-8.0]),
                np.array([8.0]))
                )
            problem.addStage(stage)

        q_init = np.linspace(np.zeros(2), qN, N)
        us_init = [np.concatenate((np.zeros(3), 0.01 * np.ones(4)))] * (N-1)
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

with open("results/acrobot_contact.txt", 'w') as file:
    for line in res:
        file.write(f"{' '.join(line)}\n")

# 0.1
# Average number of iterations:  400.0
# Average cost:  839.7110418710809
# Average violation:  0.4820552951316643
# Successes:  0

# 0.2
# Average number of iterations:  400.0
# Average cost:  539.4411790073649
# Average violation:  0.6980029165676495
# Successes:  0

# 0.05
# Convergence failure.
# Average number of iterations:  399.83870967741933
# Average cost:  1165.6889138998554
# Average violation:  0.2886359030019276
# Successes:  1

# 0.03
# Average number of iterations:  383.9032258064516
# Average cost:  1347.9924837684196
# Average violation:  0.2116171759276446
# Successes:  2

# 0.02
# Average number of iterations:  400.0
# Average cost:  1295.7719105686285
# Average violation:  0.27289239509892066
# Successes:  0