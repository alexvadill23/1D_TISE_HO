# 1D_TISE_HO

Solving the Time-Independent Schrödinger Equation (TISE) in one dimension for the harmonic oscillator potential using **Physics-Informed Neural Networks (PINNs)**.  
We implement a general code capable of obtaining the wave function for any quantum state $n$, and compare two different mesh generation methods for training points.

## Repository Contents

- `pinn.py`: Defines the architecture of the PINN.
- `1d_adap.py`: Trains the model using an adaptive mesh based on the residual; the final model is saved.
- `1d_fija.py`: Trains the model using a uniform, fixed mesh; the final model is saved.
- `1d_eval.py`: Based on the saved models for both adaptive and fixed methods, metrics are calculated and plots are generated.
- `README.md`: General documentation for the project.

## Code Functionality

All files must be in the same folder so that imports of the architecture and the saved models for evaluation work correctly.  

It is important to note that, if we want to see the solution for $n = 3$, for example, we must first implement the methods for $n = 0$, $n = 1$, and $n = 2$, because the code uses the saved models of the previous states to ensure orthogonality.

An early stopping method based on the stabilization of energy is applied, since the energy is defined as a trainable parameter and is initialized far from the true value. This allows the PINN to learn both the wave function and the corresponding energy eigenvalue. We also compare the impact of using an adaptive versus a fixed mesh.

## One-dimensional Time-Independent Schrödinger Equation for the Harmonic Oscillator

The Schrödinger equation:

$$
\frac{\hbar^2}{2m} \psi''(x) + V(x) \psi(x) = E \psi(x)
$$

For the harmonic oscillator potential:

$$
V(x) = \frac{1}{2} m \omega^2 x^2
$$

In dimensionless units ($\hbar = m = \omega = 1$), the equation becomes:

$$
\frac{1}{2} \psi''(x) + \frac{1}{2} x^2 \psi(x) = E \psi(x)
$$

The analytical eigenfunctions are:

$$
\psi_n(x) = \frac{1}{\sqrt{2^n n!}} \frac{1}{\pi^{1/4}} e^{-x^2/2} H_n(x)
$$

with eigenvalues:

$$
E_n = n + \frac{1}{2}
$$

where $H_n(x)$ are the Hermite polynomials.

The PINN learns both $\psi(x)$ and the corresponding eigenvalue $E$. The quantum state $n$ can be selected in the code to solve for the desired eigenfunction.

## Architecture and Hyperparameters

For both adaptive and fixed mesh cases, the following architecture and hyperparameters are used:

| Parameter             | Value         |
|-----------------------|---------------|
| Number of layers      | ...           |
| Neurons per layer     | ...           |
| Activation function   | ...           |
| Learning rate         | ...           |
| Optimizer             | ...           |
| Number of epochs      | ...           |

## Loss Function

The loss function used combines multiple contributions:

$$
\mathcal{L} = \lambda_{\text{PDE}} L_{\text{PDE}} + \lambda_{\text{BC}} L_{\text{BC}} + \lambda_{\text{norm}} L_{\text{norm}} + \lambda_{\text{ortho}} L_{\text{ortho}} + \lambda_{\text{parity}} L_{\text{parity}}
$$

Where:

- $L_{\text{PDE}}$ is the residual of the Schrödinger equation evaluated at collocation points.  
- $L_{\text{BC}}$ enforces the boundary conditions ($\psi(\pm x_\text{max}) = 0$).  
- $L_{\text{norm}}$ ensures wave function normalization ($\int |\psi(x)|^2 dx = 1$).  
- $L_{\text{ortho}}$ enforces orthogonality with previously computed eigenstates for $n > 0$.  
- $L_{\text{parity}}$ enforces the correct parity of the wave function (even/odd) according to the quantum number $n$.

Each term can be weighted differently using $\lambda$ coefficients to balance their contributions during training.

## Results

Results are generated and saved in `1d_eval.py` after training each state for both adaptive and fixed mesh methods. Metrics and plots include:

- Wave function comparison with analytical solution.  
- Energy convergence plots.  
- Residual and loss evolution.
