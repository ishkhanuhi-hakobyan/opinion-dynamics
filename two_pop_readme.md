 ## Class Overview
The `TwoPopulationMFG` class is designed to simulate and solve a two-population mean field game (MFG). This class models the dynamics of two distinct populations of agents, each influenced by their preferences and interactions with other agents. The primary goal is to determine the optimal strategies (control functions) and the evolution of the agent distributions over time, guided by the principles of mean field game theory.

### Key Features
- **Dual Population Dynamics**: Handles the complexities of two interacting populations, each with unique characteristics and interaction dynamics.
- **Agent Interaction Modeling**: Utilizes local interaction kernels to model how agents from different populations influence each other based on proximity and other factors.
- **Optimal Control and Distribution Evolution**: Solves for the optimal control strategies using the Hamilton-Jacobi-Bellman (HJB) approach and predicts the evolution of population distributions using the Fokker-Planck (FP) equations.
- **Numerical Methods Implementation**: Employs advanced numerical methods such as Jacobi matrices, vectorized operations, and iterative solvers to efficiently handle the complexities of MFGs.
- **Visualization and Analysis**: Provides tools and methods for visualizing the outcomes of the simulations, helping in the analysis of how different strategies and parameters affect the overall system dynamics.


## Class Methods Description
- **`__init__`**: Initializes the model with simulation parameters for two populations, including the terminal time, number of intervals, boundaries, viscosity, susceptibility coefficients, interaction strengths, interaction ranges, and agent closeness.
- **`mu0`**: Determines the initial distribution of agents for each population based on specified indices. Can be configured to provide uniform or normal distributions.
- **`uT`**: Calculates the final value function for each population, emphasizing individual goals related to specified decision points `x_d1` and `x_d2`.
- **`local_kernel`**: Computes a simple proximity kernel function that returns 1 if agents are within an epsilon distance, otherwise 0.
- **`psi`**: Calculates interaction effects between agents based on the local kernel, considering the distribution of agents and specific interaction parameters.
- **`prolong`**: Reorganizes vector data into matrices and sets boundary conditions for simulation, useful in numerical methods to ensure data consistency across time steps.
- **`K_d`**: A kernel function that measures interactions between different populations based on distance and predefined interaction strengths.
- **`G_m`**: Computes the interaction term for the Fokker-Planck equation, integrating the effects of multiple populations on individual agents.
- **`g`**: Determines the control cost function incorporating interaction terms, used in the Hamiltonian for determining optimal strategies.
- **`hamilton`**: Calculates the Hamiltonian for the system, using derivatives of the value function and control costs to find optimal policies.
- **`hjb`**: Defines the Hamilton-Jacobi-Bellman equation, central to finding optimal control strategies over time.
- **`fp_linearized_part`**: Linearizes the Fokker-Planck equation to facilitate its numerical solution, focusing on agent dynamics influenced by the value function gradients.
- **`solve_hjb`**: Solves the Hamilton-Jacobi-Bellman equation using iterative methods to approach an optimal control policy.
- **`fp`**: Solves the Fokker-Planck equation, describing the evolution of the probability distribution of agents, accounting for dynamics influenced by the optimal control found via HJB.
- **`solve_fp`**: Implements a solver for the Fokker-Planck equation, crucial for updating agent distributions based on derived controls.
- **`hjb_sys` and `fp_sys`**: System-wide methods that apply the HJB and FP solutions across all discretized time and space points.
- **`hjb_sys_vec` and `fp_sys_vec`**: Vectorized implementations of the system-wide HJB and FP methods, optimizing computation for large-scale numerical methods.
- **`solve`**: Orchestrates the full mean field game solution, alternating between solving the HJB and FP equations for both populations until convergence is achieved.