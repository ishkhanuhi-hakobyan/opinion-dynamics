import jax.numpy as jnp
import numpy as np
from jax import jacfwd, vmap
from jax.scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt
import munch
import os

x_d1 = -2
x_d2 = 2
os.environ["JAX_TRACEBACK_FILTERING"]="off"
os.environ["JAX_PLATFORM_NAME"] = 'cpu'
class TwoPopulationMFG(object):
    def __init__(self, T, Nt, xl, xr, N, nu, alphas, sigmas, lambdas, eps):
        self.T = T
        self.Nt = Nt
        self.xl = xl
        self.xr = xr
        self.N = N
        self.nu = nu
        self.alphas = alphas
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.dt = T / (Nt-1)
        self.x_grid = jnp.linspace(xl, xr, N)
        self.t_grid = jnp.linspace(0, T, Nt)
        self.eps = eps
        self.h = round((xr - xl) / (N-1), 2)

    def mu0(self, x, population_index):
        a, b = -5, 5
        arr = np.array(x)  # Convert x to a numpy array

        filtered_x = arr[(arr >= a) & (arr <= b)]

        if population_index == 0:
            midpoint = -0.5
            sigma_mu = 1
        elif population_index == 1:
            midpoint = 0.5
            sigma_mu = 1
        else:
            raise ValueError("Invalid population_index. Must be 0 or 1.")

        b1 = (a - midpoint) / sigma_mu
        b2 = (b - midpoint) / sigma_mu

        pdf_values = truncnorm.pdf(filtered_x, a=b1, b=b2, loc=midpoint, scale=sigma_mu)

        mu = np.zeros_like(arr)
        mu[(arr >= a) & (arr <= b)] = pdf_values

        return mu

    def uT(self, x, population_index):
        if population_index == 0:
            return (x - x_d1) ** 2  # For population 1
        elif population_index == 1:
            return (x - x_d2) ** 2  # For population 2*

    def local_kernel(self, x, y):
        dist_squared = (x - y) ** 2
        epsilon_squared = self.eps ** 2
        res = jnp.exp(1 - epsilon_squared / (epsilon_squared - dist_squared + 1e-15))
        return res

    def psi(self, Xtk, y, lambda_r, mu_k_t, mu, sigma_k):
        numerator = self.local_kernel(Xtk, y)

        denominator = 0.0
        kernel_values = self.local_kernel(Xtk, y)
        for r, lambda_r in enumerate(self.lambdas):
            K_d_value = self.K_d(mu_k_t, mu[r], sigma_k)


            integral_approx = jnp.sum(jnp.multiply(kernel_values, mu[r]) * self.h)

            denominator += lambda_r * K_d_value * integral_approx

        return numerator / (denominator + 1e-6)


    def prolong(self, Uvec, Mvec, idx):
        Umtx = jnp.reshape(Uvec, (self.Nt, self.N))
        Mmtx = jnp.reshape(Mvec, (self.Nt, self.N))

        U = jnp.zeros((self.Nt + 1, self.N))
        M = jnp.zeros((self.Nt + 1, self.N))

        U = U.at[:self.Nt, :].set(Umtx)
        M = M.at[1:, :].set(Mmtx)

        M0 = self.mu0(self.x_grid, idx)  # Assuming mu0 returns an array of shape (N,)
        UT = vmap(self.uT, in_axes=(0, None))(self.x_grid, idx)
        U = U.at[self.Nt, :].set(UT)
        M = M.at[0, :].set(M0)
        return U, M

    def wasserstein_1(self, points1, points2, weights1=None, weights2=None):
        n, = points1.shape
        m,  = points2.shape

            # Default to uniform weights if not provided
        if weights1 is None:
            weights1 = jnp.ones(n) / n
        if weights2 is None:
            weights2 = jnp.ones(m) / m

        # Compute cost matrix
        cost_matrix = jnp.abs(points1[:, None] - points2[None, :])

        # Assign each row to the column with the minimum cost (greedy assignment)
        col_ind = jnp.argmin(cost_matrix, axis=1)
        row_ind = jnp.arange(cost_matrix.shape[0])

        # Compute Wasserstein distance
        distance = jnp.sum(cost_matrix[row_ind, col_ind] * jnp.minimum(weights1[row_ind], weights2[col_ind]))
        return distance

    def K_d(self, x, y, sigma):
        w_p = self.wasserstein_1(x, y)
        kernel_values = jnp.where(w_p <= sigma, 1, 0)

        return kernel_values

    def G_m(self, x, mu, population_index):
        alpha_k = self.alphas[population_index]
        sigma_k = self.sigmas[population_index]

        interaction_term = 0.0

        for r, lambda_r in enumerate(self.lambdas):
            K_d_value = self.K_d(mu[population_index], mu[r], sigma_k)

            psi_values = vmap(
                lambda y: self.psi(x, y, lambda_r, mu[population_index], [mu[population_index], mu[r]], sigma_k) * y *
                          mu[r]
            )(self.x_grid)

            integral_approximation = jnp.sum(psi_values) * self.h

            interaction_term += lambda_r * integral_approximation * K_d_value

        # Final computation
        G_M_value = alpha_k * x - alpha_k * interaction_term

        return G_M_value

    def G_m2(self, x, mu, population_index):
        alpha_k = self.alphas[population_index]
        sigma_k = self.sigmas[population_index]

        interaction_term = 0.0

        for r, lambda_r in enumerate(self.lambdas):
            K_d_value = self.K_d(mu[population_index], mu[r], sigma_k)

            psi_values = vmap(
                lambda y: self.psi(x, y, lambda_r, mu[population_index], [mu[population_index], mu[r]], sigma_k) * y *
                          mu[r]
            )(self.x_grid)
            integral_approximation = jnp.sum(psi_values) * self.h

            interaction_term += lambda_r * integral_approximation * K_d_value

        # Final computation
        G_M_value = alpha_k * x - alpha_k * interaction_term

        return G_M_value

    def g(self, x, q1, q2, m1, m2, idx):
        p1 = jnp.minimum(q1, 0)
        p2 = jnp.maximum(q2, 0)
        g_m = self.G_m(x, [m1, m2], idx)

        b1 = jnp.minimum(g_m, 0)
        b2 = jnp.maximum(g_m, 0)

        return (p1 ** 2) / 2 + (p2 ** 2) / 2 + b1 * q1 + b2 * q2

    def hamilton(self, U, M1, M2, idx):
        dUR = (jnp.roll(U, -1) - U)/self.h
        dUL = (U - jnp.roll(U, 1))/self.h
        Hamiltonian = vmap(lambda x, q1, q2: self.g(x, q1, q2, M1, M2, idx))(self.x_grid, dUR, dUL)

        return Hamiltonian

    def hjb(self, ts, Uk1, Uk, Mk1, Mk2, idx):
        UR = jnp.roll(Uk, -1)
        UL = jnp.roll(Uk, 1)

        Delta_U = - (2 * Uk - UR - UL) / (self.h ** 2)

        Dt_U = (Uk1 - Uk) / self.dt

        Hamiltonian = self.hamilton(Uk, Mk1, Mk2, idx)

        return -Dt_U - self.nu * Delta_U + Hamiltonian - Mk1

    def fp_linearized_part(self, U, M1, M2, idx):
        UR = (jnp.roll(U, -1) - U) / self.h
        UL = (U - jnp.roll(U, 1)) / self.h

        URF = UR.flatten()
        ULF = UL.flatten()

        dGq1 = lambda x1, q1, q2: jnp.minimum(q1, 0)
        dGq2 = lambda x1, q1, q2: jnp.maximum(q2, 0)

        dGq1s = vmap(dGq1)(self.x_grid, URF, ULF)
        dGq2s = vmap(dGq2)(self.x_grid, URF, ULF)

        dGq1s = jnp.multiply(dGq1s, M1)
        dGq2s = jnp.multiply(dGq2s, M1)

        dGqDifference1 = dGq1s - dGq2s

        dGqs2R = jnp.roll(dGq2s, -1)
        dGqs1L = jnp.roll(dGq1s, 1)

        A = - (dGqDifference1 + dGqs2R - dGqs1L) / self.h

        b = vmap(lambda x: self.G_m2(x, [M1, M2], idx))(self.x_grid)
        bp1 = lambda x: jnp.minimum(x, 0)
        bp2 = lambda x: jnp.maximum(x, 0)

        bp1s = vmap(bp1)(b)
        bp2s = vmap(bp2)(b)

        bp1s = jnp.multiply(bp1s, M1)
        bp2s = jnp.multiply(bp2s, M1)

        bpDifference1 = bp1s - bp2s

        bp2sR = jnp.roll(bp2s, -1)
        bp1sL = jnp.roll(bp1s, 1)

        B = - (bpDifference1 + bp2sR - bp1sL) / self.h

        MEQByHands = A + B

        return MEQByHands


    def solve_hjb(self, U0, M, lr, idx, tol=10**(-7), epoch=50):
        error = 1
        iter_num = 0

        U = U0[idx]
        U_i = U0
        while error > tol and iter_num < epoch:
            b = self.hjb_sys_vec(U_i, M, idx)
            jacobi = jacfwd(self.hjb_sys_vec)(U_i, M, idx)

            dz = jnp.linalg.solve(jacobi[idx], b.flatten())
            U = U - lr * dz
            error = jnp.dot(dz, dz)
            U_i[idx] = U
            print("             the error of solving hjb is {}".format(error))

            iter_num = iter_num + 1

        return U


    def fp(self, Uk, Mk1, Mk, M2, idx):
        Dt_M = (Mk1 - Mk) / self.dt

        Mk1R = jnp.roll(Mk1, -1)
        Mk1L = jnp.roll(Mk1, 1)

        Delta_M = - (2 * Mk1 - Mk1R - Mk1L) / self.h**2

        adj = self.fp_linearized_part(Uk, Mk1, M2, idx)

        return Dt_M - self.nu * Delta_M + adj

    def solve_fp(self, M0, U1, idx):
        jacobi = jacfwd(self.fp_sys_vec)(M0, U1, idx)
        Mvec = [jnp.zeros_like(M0[0]), M0[1]] if idx == 0 else [M0[0], jnp.zeros_like(M0[1])]
        b = self.fp_sys_vec(Mvec, U1, idx)
        M = jnp.linalg.lstsq(jacobi[idx], -b)

        return M[0]

    def hjb_sys(self, U, M1, M2, idx):
        ts = jnp.linspace(0, self.T, self.Nt+1)
        if idx ==0:
            hjbs = vmap(self.hjb, in_axes=(0, 0, 0, 0, 0, None))(ts[:-1], U[1:, :], U[:-1, :], M1[1:, :], M2[1:, :], idx)
            return hjbs
        if idx == 1:
            hjbs = vmap(self.hjb, in_axes=(0, 0, 0, 0, 0, None))(ts[:-1], U[1:, :], U[:-1, :], M2[1:, :], M1[1:, :], idx)
            return hjbs

    def fp_sys(self, U, M1, M2, idx):
        fps = vmap(self.fp, in_axes=(0, 0, 0, 0, None))(U[:-1, :], M1[1:, :], M1[:-1, :], M2[1:, :], idx)
        return fps

    def hjb_sys_vec(self, Uvec, Mvec, idx):
        U1, M1 = self.prolong(Uvec[0], Mvec[0], 0)
        U2, M2 = self.prolong(Uvec[1], Mvec[1], 1)
        if idx == 0:
            return self.hjb_sys(U1, M1, M2, idx).flatten()
        if idx == 1:
            return self.hjb_sys(U2, M1, M2, idx).flatten()

    def fp_sys_vec(self, Mvec, Uvec, idx):
        U1, M1 = self.prolong(Uvec[0], Mvec[0], 0)
        U2, M2 = self.prolong(Uvec[1], Mvec[1], 1)
        if idx == 0:
            return self.fp_sys(U1, M1, M2, idx).flatten()
        if idx == 1:
            return self.fp_sys(U2, M2, M1, idx).flatten()


    def solve(self, tol=10**(-6), epoch=100, hjb_lr=1, hjb_epoch=100):
        U1 = jnp.zeros((self.Nt, self.N)).flatten()
        U2 = jnp.zeros((self.Nt, self.N)).flatten()
        M1 = jnp.zeros((self.Nt, self.N)).flatten()
        M2 = jnp.zeros((self.Nt, self.N)).flatten()

        error = 1
        iter_num = 0

        while error > tol and iter_num < epoch:
            new_U1 = self.solve_hjb([U1, U2], [M1, M2], hjb_lr, epoch=hjb_epoch, idx=0)
            new_U2 = self.solve_hjb([U1, U2], [M1, M2], hjb_lr, epoch=hjb_epoch, idx=1)
            new_M1 = self.solve_fp([M1, M2], [new_U1, new_U2], 0)
            print("herw2")
            new_M2 = self.solve_fp([M1, M2], [new_U1, new_U2], 1)

            U_err = jnp.dot(new_U1 - U1, new_U1 - U1) + jnp.dot(new_U2 - U2, new_U2 - U2)
            M_err = jnp.dot(new_M1 - M1, new_M1 - M1) + jnp.dot(new_M2 - M2, new_M2 - M2)
            error = U_err + M_err

            print('MFG error: {}'.format(error))

            U1, U2, M1, M2 = new_U1, new_U2, new_M1, new_M2
            iter_num += 1
            print("Iteration: {}".format(iter_num))

        U1, M1 = self.prolong(U1, M1, 0)
        U2, M2 = self.prolong(U2, M2, 1)

        return U1, M1, U2, M2


cfg = munch.munchify({
    'T' : 2,
    'Nt': 31,
    'xl': -6,
    'xr': 6,
    'N' : 71,
    'nu': 0.5,
    'alphas': [0.01, 0.01],
    'sigmas': [0.01, 0.01],
    'lambdas': [0.5, 0.5],
    'eps': 0.5,
    'hjb_epoch': 50,
    'hjb_lr': 0.5,
    'epoch': 50,
    'lr': 1,
    'tol': 10 ** (-5),
})


solver = TwoPopulationMFG(T=cfg.T, Nt=cfg.Nt, xl=cfg.xl, xr=cfg.xr, N=cfg.N, nu=cfg.nu, alphas=cfg.alphas, sigmas=cfg.sigmas, lambdas=cfg.lambdas, eps=cfg.eps)


TT = jnp.linspace(0, cfg.T, cfg.Nt)
XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N)
TT, XX = jnp.meshgrid(TT, XX)

# U, M = solver.solve_mfg(cfg.lr, cfg.tol, cfg.epoch)
U1, M1, U2, M2 = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)


 # Assuming cfg, U1, U2, M1, M2, x_d1, and x_d2 are already defined
TT = np.linspace(0, cfg.T, cfg.Nt)
XX = np.linspace(-10, 10, cfg.N)
max_idx1 = np.argmax(M1[-1, :]).item()
max_init1= np.argmax(M1[0, :]).item()
max_item1 = XX[max_idx1]
max_item_init1= XX[max_init1]
final_mean1 = max_item1
init_mean1 = max_item_init1

max_idx2 = np.argmax(M2[-1, :]).item()
max_init2= np.argmax(M2[0, :]).item()
max_item2 = XX[max_idx2]
max_item_init2= XX[max_init2]
final_mean2 = max_item2
init_mean2 = max_item_init2
# Final value function for both populations
plt.figure()
plt.plot(XX, U1[-1, :], label='U1(T)')
plt.plot(XX, U2[-1, :], label='U2(T)')
plt.title('Final Value Function')
plt.xlabel('x')
plt.ylabel('u(T)')
plt.legend()
plt.show()

plt.plot(XX, M1[0, :], label=f'Mean: {round(init_mean1, 2)}')  # More descriptive name for M1(T)
plt.plot(XX, M2[0, :], label=f'Mean: {round(init_mean2, 2)}')  # More descriptive name for M2(T)
plt.xlabel('x')
plt.ylabel('m(T)')
plt.title(r"$\alpha=" +f"{cfg.alphas}" +r",\ \varepsilon=" +f"{cfg.eps}, T={cfg.T}$")
plt.legend(loc='upper left')  # Display the legend with subscript notation
plt.savefig('Initial_Distribution.png', format='png', dpi=300)
plt.show()

# Final distribution for both populations with updated legends
plt.figure()
plt.plot(XX, M1[-1, :], label=f'Mean: {round(final_mean1, 2)}')  # More descriptive name for M1(T)
plt.plot(XX, M2[-1, :], label=f'Mean: {round(final_mean2, 2)}')  # More descriptive name for M2(T)
plt.axvline(x=x_d1, color='Blue', linestyle='--', linewidth=1, label=r'Desired Opinion')  # LaTeX for subscript
plt.axvline(x=x_d2, color='Red', linestyle='--', linewidth=1, label=r'Desired Opinion')  # LaTeX for subscript
plt.xlabel('x')
plt.ylabel('m(T) final')
plt.title(r"$\alpha=" +f"{cfg.alphas}" +r",\ \varepsilon=" +f"{cfg.eps}, T={cfg.T}$")
plt.legend(loc='upper left')  # Display the legend with subscript notation
plt.savefig('Final_Distribution.png', format='png', dpi=300)
plt.show()
