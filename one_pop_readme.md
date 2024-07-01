# README for OnePopulationalMFG Code

## Description
This Python module implements a numerical solver for a one-populational mean field game (MFG). The model simulates the dynamics of agents whose behavior is influenced by both their individual preferences and the overall population distribution.

## Features
- Computation of Hamiltonian for control problems using backward and forward equations.
- Utilization of JAX for automatic differentiation and efficient array manipulation.
- Visualization of results using Matplotlib for scalar fields over time and final distributions.

## Dependencies
- JAX
- JAX NumPy
- NumPy
- Matplotlib
- Munch

To install dependencies, run:
```bash
pip install jax jaxlib numpy matplotlib munch
```

## Configuration

The model is configured using a munch.Munch dictionary which allows easy access using dot notation. It includes parameters like terminal time T, number of time intervals Nt, spatial boundaries xl and xr, and other parameters relevant to the dynamics.


## Class Methods Description
- **`__init__`**: Initializes the model with simulation parameters such as the terminal time, number of intervals, boundaries, viscosity, susceptibility, and agent closeness.
- **`m0`**: Defines the initial distribution of agents across the space.
- **`uT`**: Represents the final value function, expressing each agent's preference at the terminal time.
- **`r`**: Calculates the normalization factor for the interaction term based on the entire population's distribution.
- **`phi`**: Defines the pairwise interaction function, which dictates how agents influence each other based on their distance.
- **`b`**: Computes the drift term in the mean field game, representing the net movement tendency of agents.
- **`g`**: Calculates the running cost associated with a given control and state.
- **`hamilton`**: Computes the Hamiltonian of the system using the value function derivatives.
- **`fp_linearized_part`**: Linearizes the Fokker-Planck equation to facilitate numerical solution.
- **`hjb`**: Represents the backward Hamilton-Jacobi-Bellman equation in the optimization.
- **`fp`**: Represents the forward Fokker-Planck equation describing the evolution of the density of agents.
- **`hjb_sys` and `fp_sys`**: Vectorized versions of the HJB and FP equations to handle system-wide calculations.
- **`prolong`**: Extends vectors to matrices for handling boundary conditions in the numerical scheme.
- **`hjb_sys_vec` and `fp_sys_vec`**: Converts system-wide HJB and FP calculations to vectorized forms for use in numerical methods.
- **`solve_hjb` and `solve_fp`**: Solvers for the HJB and FP equations, using iterative numerical methods.
- **`solve`**: Orchestrates the entire solution process, iterating between solving the HJB and FP equations until convergence.