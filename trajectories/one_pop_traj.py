import numpy as np
import matplotlib.pyplot as plt


class BoundedConfidenceModel:
    def __init__(self, n, bnd, noisy_std, alpha, eps, nu):
        """
        Initialize the population model for bounded confidence dynamics.

        Parameters:
        n: int, number of particles
        bnd: float, boundary for the state space
        noisy_std: float, standard deviation for Gaussian noise
        alpha: array-like, susceptibility parameters for each particle
        eps: array-like, confidence thresholds for each particle
        nu: float, diffusion parameter for the Wiener process
        """
        self.n = n
        self.bnd = bnd
        self.alpha = np.array(alpha)
        self.eps = np.array(eps)
        self.nu = nu
        self.x = None
        self.dt = 0.1

    def phi(self, x_i, x_j, epsilon):
        """
        Weighting function based on the bounded confidence model.
        Returns 1 if agents are within the confidence bounds, otherwise 0.
        """
        return 1.0 if abs(x_i - x_j) < epsilon else 0.0

    def A_i(self, x_i, x_, eps_i):
        """
        Normalizing factor A^i(t).
        """
        return sum([self.phi(x_i, x_j, eps_i) for x_j in x_])

    def dyn(self, x, u=None):
        """
        Update the state of the system based on control input and bounded confidence dynamics.

        Parameters:
        x: list of current states
        u: list of control inputs for each state (if any)
        """
        x_ = np.copy(x)
        new_x = np.zeros_like(x_)

        for i in range(self.n):
            A_i_t = self.A_i(x_[i], x_, self.eps[i])

            # Compute the interaction term
            interaction_sum = sum(
                self.phi(x_[i], x_[j], self.eps[i]) * x_[j]
                for j in range(self.n) if i != j
            )

            # Dynamics with control input and Wiener process
            new_x[i] = -self.alpha[i] * x_[i] + (self.alpha[i] / A_i_t) * interaction_sum
            if u is not None:
                new_x[i] += u[i] * self.dt

        # Noise term: Wiener process
        if self.nu is not None:
            noise = np.random.normal(0, np.sqrt(2 * self.nu * self.dt), size=x_.shape)
            new_x += noise

        new_x = np.clip(new_x, -self.bnd, self.bnd)

        return new_x

    def update(self, u=None):
        """
        Update the state of the population.
        """
        self.x = self.dyn(self.x, u=u)

    def visualize(self, iteration=None):
        """
        Visualize the current state of the population in 1D.
        """
        states = np.array(self.x)
        plt.scatter(states, np.zeros_like(states), c='b', label='Population States')
        plt.xlim(-self.bnd, self.bnd)
        plt.ylim(-0.5, 0.5)  # Flatten y-axis
        plt.title(f'Population Dynamics {f"Iteration {iteration}" if iteration is not None else ""}')
        plt.legend()
        plt.show()


n = 50
bnd = 5.0
alpha = np.random.uniform(0, 1, n)  # Random susceptibility for each agent
eps = np.random.uniform(0.5, 1.5, n)  # Random confidence bounds for each agent
nu = 0.05  # Diffusion parameter for noise

mean = 0.0
std = 1.0
x_initial = np.random.normal(mean, std, n)

model = BoundedConfidenceModel(n=n, bnd=bnd, noisy_std=0.1, alpha=alpha, eps=eps, nu=nu)
model.x = x_initial

for i in range(10):
    model.visualize(iteration=i)
    model.update(u=None)

model.visualize(iteration="Final")