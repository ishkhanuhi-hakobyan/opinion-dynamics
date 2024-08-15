import numpy as np
import matplotlib.pyplot as plt

class One2DTraj:
    def __init__(self, n, bnd, noisy_std, alpha):
        """
        Initialize the population model.

        Parameters:
        n: int, number of particles
        bnd: float, boundary for the state space
        noisy_std: float, standard deviation for Gaussian noise
        alpha: float, parameter for convex combination with previous state
        """
        self.n = n
        self.bnd = bnd
        self.std = noisy_std
        self.alpha = alpha
        self.x = None
        self.dt = 0.1

    def dyn(self, x, u=None):
        """
        Update the state of the system based on the control input.

        Parameters:
        x: list of tuples, current states and their sources
        u: list of np arrays, control input for each state
        """
        x_ = np.array([state for state, _ in x])

        # Apply control input
        if u is not None:
            x_ += np.array(u) * self.dt

        # Optionally apply noise
        if self.std is not None:
            noise = np.random.normal(0, self.std, size=x_.shape)
            x_ += noise

        # Apply boundary clipping
        x_ = np.clip(x_, -self.bnd, self.bnd)

        return [(x_i, source_i) for x_i, (_, source_i) in zip(x_, x)]

    def update(self, u=None):
        """
        Update the state of the population.
        """
        self.x = self.dyn(self.x, u=u)

    def visualize(self, iteration=None):
        """
        Visualize the current state of the population.
        """
        states = np.array([x_i for x_i, _ in self.x])
        plt.scatter(states[:, 0], states[:, 1], c='b', label='Population States')
        plt.xlim(-self.bnd, self.bnd)
        plt.ylim(-self.bnd, self.bnd)
        plt.title(f'Population Dynamics {f"Iteration {iteration}" if iteration is not None else ""}')
        plt.legend()
        plt.show()


# Usage example
n = 50  # Number of particles
bnd = 5.0
alpha = 0.5
noisy_std = 0.1

# Mean and covariance for the 2D normal distribution
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.0], [0.0, 1.0]])

# Generate initial states from a 2D normal distribution
x_initial = [(np.random.multivariate_normal(mean, cov), 'population') for i in range(n)]
model = One2DTraj(n=n, bnd=bnd, noisy_std=noisy_std, alpha=alpha)
model.x = x_initial

# Visualization over iterations
for i in range(10):
    model.visualize(iteration=i)
    model.update(u=None)

model.visualize(iteration="Final")