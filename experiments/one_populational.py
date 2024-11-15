import jax
from jax import jacfwd, jit
import jax.numpy as jnp
from jax import vmap, lax
from jax.scipy.stats import norm, truncnorm
import numpy as np
import matplotlib.pyplot as plt
import munch
import os

from matplotlib.lines import Line2D

os.environ["JAX_TRACEBACK_FILTERING"]="off"



x_d = -2
class OnePopulationalMFG(object):
    def __init__(self, T, Nt, xl, xr, N, nu, alpha, eps):
        """
        Input:

        T:          float
                the terminal time

        Nt:         integer
                the number of time intervals

        xl:         float
                the left boundary of the interval

        xr:         float
                the right boundary of the interval

        N:          integer
                the number of intervals

        nu:         float
                the viscosity constant

        alpha:      float
                the coefficient of the susceptibility of the population

        eps:        float
                define the closeness of agents
        """
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt
        self.xl = xl
        self.xr = xr
        self.N = N
        self.nu = nu
        self.alpha = alpha
        self.X = jnp.linspace(xl, xr, N)
        self.h = (xr - xl) / (N-1)
        self.eps = eps


    def m0(self, x):
        a, b = -5, 5
        arr = np.array(x)
        filtered_x = np.where(arr>=a, arr, 0)
        # x = np.where(filtered_x<=b, filtered_x, 0)
        midpoint = 1
        sigma_mu = 1
        b1 = (a - midpoint) / sigma_mu
        b2 = (b - midpoint) / sigma_mu
        mu = truncnorm.pdf(x, a=b1, b=b2, loc=midpoint, scale=sigma_mu)

        return mu

    def uT(self, x):
        return (x - x_d) ** 2  # we suppose that every agent prefers to zero opinion

    def r(self, x, m):
        return 1 / (jnp.sum(vmap(lambda y: self.phi(x, y))(self.X)*m) +1e-10)/self.N

    def phi(self, x, y):
        dist = jnp.minimum(jnp.abs(x - y), self.eps)
        res = jnp.exp(1 - self.eps**2 / (1e-40 + self.eps**2 - dist**2))  #add a small number to prevent from dividing by zero
        return res

    def b(self, x, m):
        interaction = jnp.sum(vmap(lambda y: self.phi(x, y))(self.X) * self.X * m) / self.N

        return self.alpha* x - self.alpha * self.r(x, m) * interaction

    def g(self, x, q1, q2, m):
        p1 = jnp.minimum(q1, 0)
        p2 = jnp.maximum(q2, 0)
        b = self.b(x, m)
        b1 = jnp.minimum(b, 0)
        b2 = jnp.maximum(b, 0)
        return (p1 ** 2) / 2 + (p2 ** 2) / 2 + b1 * q1 + b2 * q2

    def hamilton(self, U, M):
        dUR = (jnp.roll(U, -1) - U)/self.h
        dUL = (U - jnp.roll(U, 1))/self.h

        Hamiltonian = vmap(lambda x, q1, q2: self.g(x, q1, q2, M))(self.X, dUR, dUL)
        return Hamiltonian

    def fp_linearized_part(self, U, M):
        UR = (jnp.roll(U, -1) - U) / self.h
        UL = (U - jnp.roll(U, 1)) / self.h

        URF = UR.flatten()
        ULF = UL.flatten()

        dGq1 = lambda x1, q1, q2: jnp.minimum(q1, 0)
        dGq2 = lambda x1, q1, q2: jnp.maximum(q2, 0)

        dGq1s = vmap(dGq1)(self.X, URF, ULF)
        dGq2s = vmap(dGq2)(self.X, URF, ULF)

        dGq1s = jnp.multiply(dGq1s, M)
        dGq2s = jnp.multiply(dGq2s, M)

        dGqDifference1 = dGq1s - dGq2s

        dGqs2R = jnp.roll(dGq2s, -1)
        dGqs1L = jnp.roll(dGq1s, 1)

        A = - (dGqDifference1 + dGqs2R - dGqs1L) / self.h

        # consider the effect of b(x)
        b = vmap(lambda x: self.b(x, M))(self.X)
        bp1 = lambda x: jnp.minimum(x, 0)
        bp2 = lambda x: jnp.maximum(x, 0)

        bp1s = vmap(bp1)(b)
        bp2s = vmap(bp2)(b)

        bp1s = jnp.multiply(bp1s, M)
        bp2s = jnp.multiply(bp2s, M)

        bpDifference1 = bp1s - bp2s

        bp2sR = jnp.roll(bp2s, -1)
        bp1sL = jnp.roll(bp1s, 1)

        B = - (bpDifference1 + bp2sR - bp1sL) / self.h

        MEQByHands = A + B
        return MEQByHands

    def hjb(self, t, Uk1, Uk, Mk1):
        UR = jnp.roll(Uk, -1)
        UL = jnp.roll(Uk, 1)

        Delta_U = - (2 * Uk - UR - UL) / self.h ** 2

        Dt_U = (Uk1 - Uk) / self.dt

        Hamiltonian = self.hamilton(Uk, Mk1)

        return -Dt_U - self.nu * Delta_U + Hamiltonian - Mk1


    def fp(self, Uk, Mk1, Mk):
        Dt_M = (Mk1 - Mk) / self.dt

        Mk1R = jnp.roll(Mk1, -1)
        Mk1L = jnp.roll(Mk1, 1)

        Delta_M = - (2 * Mk1 - Mk1R - Mk1L) / self.h**2

        adj = self.fp_linearized_part(Uk, Mk1)


        return Dt_M - self.nu * Delta_M + adj

    def hjb_sys(self, U, M):
        """
        Input:

        U:      matrix
            (Nt + 1) x N matrix, each row contains the values of U at the grid points
            the first row represents the time 0 and the last row contains the data at time T

        M:      matrix
            (Nt + 1) x N matrix, each row contains the values of M at the grid points
            the first row represents the time 0 and the last row contains the data at time T
        """
        ts = jnp.linspace(0, self.T, self.Nt + 1)
        hjbs = vmap(self.hjb)(ts[:-1], U[1:, :], U[:-1, :], M[1:, :])

        return hjbs


    def fp_sys(self, U, M):
        fps = vmap(self.fp)(U[:-1, :], M[1:, :], M[:-1, :])

        return fps


    def prolong(self, Uvec, Mvec):
        Umtx = jnp.reshape(Uvec, (self.Nt, self.N))
        Mmtx = jnp.reshape(Mvec, (self.Nt, self.N))

        U = jnp.zeros((self.Nt + 1, self.N))
        M = jnp.zeros((self.Nt + 1, self.N))

        U = U.at[:self.Nt, :].set(Umtx)
        M = M.at[1:, :].set(Mmtx)

        # compute the values of m at time 0 and the values of u at time T
        M0 = self.m0(self.X)
        UT = vmap(self.uT)(self.X)

        U = U.at[self.Nt, :].set(UT)
        M = M.at[0, :].set(M0)
        return U, M

    def hjb_sys_vec(self, Uvec, Mvec):
        U, M = self.prolong(Uvec, Mvec)
        return self.hjb_sys(U, M).flatten()

    def fp_sys_vec(self, Mvec, Uvec):
        U, M = self.prolong(Uvec, Mvec)
        return self.fp_sys(U, M).flatten()

    def solve_hjb(self, U0, M, lr, tol=10**(-6), epoch=100):
        error = 1
        iter_num = 0

        U = U0
        while error > tol and iter_num < epoch:
            b = self.hjb_sys_vec(U, M)
            jacobi = jacfwd(self.hjb_sys_vec)(U, M)
            dz = jnp.linalg.solve(jacobi, b.flatten())
            U = U - lr * dz

            error = jnp.dot(dz, dz)
            print("             the error of solving hjb is {}".format(error))

            iter_num = iter_num + 1

        return U

    def solve_fp(self, M0, U):
        jacobi = jacfwd(self.fp_sys_vec)(M0, U)
        b = self.fp_sys_vec(jnp.zeros(len(jacobi)), U)
        M = jnp.linalg.lstsq(jacobi, -b)
        return M[0]

    def solve(self, tol=10**(-6), epoch=100, hjb_lr = 1, hjb_epoch = 100):
        U = jnp.zeros((self.Nt, self.N)).flatten()
        M = jnp.zeros((self.Nt, self.N)).flatten()

        error = 1
        iter_num = 0
        while error > tol and iter_num < epoch:
            U1 = self.solve_hjb(U, M, hjb_lr, epoch = hjb_epoch)
            M1 = self.solve_fp(M, U1)

            Uerr = U1 - U
            Merr = M1 - M

            error = jnp.dot(Uerr, Uerr) + jnp.dot(Merr, Merr)
            print('the mfg error is {}'.format(error))

            iter_num = iter_num + 1

            U = U1
            M = M1

        U, M = self.prolong(U, M)
        return U, M

cfg = munch.munchify({
    'T' : 2,
    'Nt': 20,
    'xl': -7,
    'xr': 7,
    'N' : 141,
    'nu': 0.5,
    'alpha': 0.01,
    'eps': 0.5,
    'hjb_epoch': 100,
    'hjb_lr': 1,
    'epoch': 100,
    'lr': 1,
    'tol' : 10 ** (-6),
})

dt = cfg.T / cfg.Nt
dx = (cfg.xr-cfg.xl)/cfg.N
if  not (2* cfg.nu * dt)<= dx**2:
    solver = OnePopulationalMFG(cfg.T, cfg.Nt, cfg.xl, cfg.xr, cfg.N, cfg.nu, cfg.alpha, cfg.eps)

    TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
    XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N)
    TT, XX = jnp.meshgrid(TT, XX)

    U, M = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)
    max_idx = np.argmax(M[-1, :]).item()
    max_init = np.argmax(M[0, :]).item()
    XX = jnp.linspace(-10, 10, cfg.N)
    max_item = XX[max_idx]
    max_item_init = XX[max_init]
    final_mean = round(max_item, 2)
    init_mean = round(max_item_init, 2)

    TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
    # Plot for U[-1, :]
    fig = plt.figure()
    plt.plot(XX, U[-1, :])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(T)$")
    plt.title("Final Value Function $u(T)$")

    fig = plt.figure()
    plt.plot(XX, M[0, :], label=f"Initial mean: {round(float(init_mean), 2)}")
    plt.plot(XX, M[-1, :], label=f"Final mean: {round(float(final_mean), 2)}")
    plt.axvline(x=x_d, color='r', linestyle='--',)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$m(T)$")
    plt.title(r"$\alpha=" + f"{cfg.alpha}" + r",\ \varepsilon=" + f"{cfg.eps}, T={cfg.T}$")
    plt.savefig("one_populational.png", dpi=300)
    plt.legend()
    plt.show()