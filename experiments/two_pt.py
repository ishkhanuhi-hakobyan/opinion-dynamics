import torch
import numpy as np
import os
import munch

x_d1 = -2
x_d2 = 2
os.environ["PYTORCH_JIT"] = "off"

class TwoPopulationMFG:
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
        self.dt = T / (Nt - 1)
        self.x_grid = torch.linspace(xl, xr, N)
        self.t_grid = torch.linspace(0, T, Nt)
        self.eps = eps
        self.h = round((xr - xl) / (N - 1), 2)

    def mu0(self, x, population_index):
        a, b = -6, 6
        filtered_x = x[(x >= a) & (x <= b)]
        midpoint = -1 if population_index == 0 else 1
        sigma_mu = 1
        b1 = (a - midpoint) / sigma_mu
        b2 = (b - midpoint) / sigma_mu

        pdf_values = torch.distributions.Normal(midpoint, sigma_mu).log_prob(filtered_x).exp()
        mu = torch.zeros_like(x)
        mu[(x >= a) & (x <= b)] = pdf_values
        return mu

    def uT(self, x, population_index):
        target = x_d1 if population_index == 0 else x_d2
        return (x - target) ** 2

    def local_kernel(self, x, y):
        dist = torch.abs(x - y)
        return torch.exp(1 - self.eps ** 2 / (1e-15 + self.eps ** 2 - dist ** 2))

    def prolong(self, Uvec, Mvec, idx):
        Umtx = Uvec.view(self.Nt, self.N)
        Mmtx = Mvec.view(self.Nt, self.N)
        U = torch.zeros(self.Nt + 1, self.N)
        M = torch.zeros(self.Nt + 1, self.N)
        U[:-1, :] = Umtx
        M[1:, :] = Mmtx
        M[0, :] = self.mu0(self.x_grid, idx)
        U[-1, :] = torch.tensor([self.uT(x, idx) for x in self.x_grid])
        return U, M

    def g(self, x, q1, q2, m, idx):
        p1 = torch.minimum(q1, torch.tensor(0.0))
        p2 = torch.maximum(q2, torch.tensor(0.0))
        g_m = self.G_m(x, m, idx)
        b1 = torch.minimum(g_m, torch.tensor(0.0))
        b2 = torch.maximum(g_m, torch.tensor(0.0))
        return (p1 ** 2) / 2 + (p2 ** 2) / 2 + b1 * q1 + b2 * q2

    def hamilton(self, U, M, idx):
        dUR = (torch.roll(U, shifts=-1, dims=1) - U) / self.h
        dUL = (U - torch.roll(U, shifts=1, dims=1)) / self.h
        Hamiltonian = torch.stack([self.g(x, q1, q2, M, idx) for x, q1, q2 in zip(self.x_grid, dUR.T, dUL.T)], dim=0)
        return Hamiltonian

    def solve(self, tol=1e-6, epoch=100, hjb_lr=1, hjb_epoch=100):
        U1 = torch.zeros((self.Nt, self.N)).flatten()
        U2 = torch.zeros((self.Nt, self.N)).flatten()
        M1 = torch.zeros((self.Nt, self.N)).flatten()
        M2 = torch.zeros((self.Nt, self.N)).flatten()
        error = 1
        iter_num = 0

        while error > tol and iter_num < epoch:
            # Placeholder functions for HJB and FP solvers
            new_U1 = U1  # Update logic for HJB solver
            new_U2 = U2
            new_M1 = M1  # Update logic for FP solver
            new_M2 = M2

            U_err = torch.sum((new_U1 - U1) ** 2) + torch.sum((new_U2 - U2) ** 2)
            M_err = torch.sum((new_M1 - M1) ** 2) + torch.sum((new_M2 - M2) ** 2)
            error = U_err + M_err
            print(f"MFG error: {error.item()}")

            U1, U2, M1, M2 = new_U1, new_U2, new_M1, new_M2
            iter_num += 1
            print(f"Iteration: {iter_num}")

        U1, M1 = self.prolong(U1, M1, 0)
        U2, M2 = self.prolong(U2, M2, 1)
        return U1, M1, U2, M2


cfg = munch.munchify({
    'T': 2,
    'Nt': 21,
    'xl': -7,
    'xr': 7,
    'N': 71,
    'nu': 1,
    'alphas': [0.5, 0.5],
    'sigmas': [0.01, 0.01],
    'lambdas': [0.5, 0.5],
    'eps': 0.5,
    'hjb_epoch': 100,
    'hjb_lr': 0.5,
    'epoch': 50,
    'lr': 0.5,
    'tol': 1e-5,
})

solver = TwoPopulationMFG(cfg.T, cfg.Nt, cfg.xl, cfg.xr, cfg.N, cfg.nu, cfg.alphas, cfg.sigmas, cfg.lambdas, cfg.eps)
U1, M1, U2, M2 = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)
