# 1D_TISE_HO
Solving the Time-Independent Schrödinger Equation (TISE) in one dimension for the harmonic oscillator potential using Physics-Informed Neural Networks (PINNs).
We implement a general code capable of obtaining the wave function for any quantum state n, and compare two different mesh generation methods for training points.

## Repository Contents:

- pinn.py: Defines the architecture of the PINN.

- 1d_adap.py: Trains the model using an adaptive mesh based on the residual; the final model is saved.

- 1d_fija.py: Trains the model using a uniform, fixed mesh; the final model is saved.

- 1d_eval.py: Based on the saved models for both adaptive and fixed methods, metrics are calculated and plots are generated.

- README.md: General documentation for the project.

## Code Functionality

All this files must be in the same folder, so that the importations of the architecture, or the saved models for the eval can be correctly done. 

## One dimensional Time Independent Schrödinger Equation for the harmonic oscillator

The Schrödinger equation:
-(ħ² / 2m) ψ''(x) + V(x) ψ(x) = E ψ(x)


For the harmonic oscillator potential:

V(x) = (1/2) m ω² x²


We work in dimensionless units (ħ = m = ω = 1), so the equation becomes:

-(1/2) ψ''(x) + (1/2) x² ψ(x) = E ψ(x)

The analytical eigenfunctions for the harmonic oscillator are given by:

ψₙ(x) = (1 / sqrt(2ⁿ n!)) * (1 / π^(1/4)) * exp(-x²/2) * Hₙ(x)


with eigenvalues:

Eₙ = n + 1/2


where Hₙ(x) are the Hermite polynomials.

The PINN learns both ψ(x) and the corresponding eigenvalue E. The quantum state n can be selected in the code to solve for the desired eigenfunction. It is important to note that, if we want to see the solution for n = 3, for example, we must first implement the methods for n = 0, n = 1 and n = 2, because the code uses the saved models of the previous states to ensure orthogonality.





