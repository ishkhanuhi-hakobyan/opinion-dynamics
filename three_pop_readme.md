## Class Overview
The `ThreePopulationMFG` class is designed to model and solve a three-population mean field game (MFG). This sophisticated simulation framework captures the dynamics of three distinct groups of agents, each characterized by specific interaction parameters, decision goals, and susceptibility to collective behavior. It aims to optimize the collective and individual strategies of these populations through numerical methods based on the principles of mean field theory.

### Key Features
- **Three Distinct Populations**: Manages the interactions and dynamics of three separate groups, providing a robust model for complex systems where multiple stakeholders or species interact.
- **Flexible Interaction Modeling**: Incorporates adjustable parameters for interaction strength, range, and individuality of agent dynamics, allowing for a detailed representation of diverse systems.
- **Optimal Strategy and Distribution Computation**: Determines optimal control strategies using the Hamilton-Jacobi-Bellman (HJB) framework and calculates the evolution of agent distributions with the Fokker-Planck (FP) equations.
- **Advanced Numerical Techniques**: Uses state-of-the-art numerical methods, including vectorized operations and Jacobian matrices, to efficiently solve complex differential equations inherent in MFGs.
- **Visualization Capabilities**: Provides mechanisms to visualize the resulting strategies and distributions, aiding in understanding and analysis of the simulation outcomes.

### Methodology
- The model initializes with defined parameters for each population, such as diffusion coefficients and interaction strengths.
- Initial and terminal conditions are set based on predefined goals and initial distributions.
- The HJB equations are solved to find optimal controls that minimize costs over time for each population.
- The FP equations use these controls to simulate the evolution of each population's distribution through time.
- The outcomes are visualized through various plots, including 3D surface plots of both the value functions and the distributions, to provide insights into the dynamics and efficacy of chosen strategies.