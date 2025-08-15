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

| Parameter                     | Value                         | Parameter                     | Value                         |
|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| Arquitectura                  | FC (64‑64‑64‑64)             | Activación                     | `tanh`                        |
| $\lambda_{\text{PDE}}$        | 0.3                           | Dominio                        | $x \in [-6,6]$                |
| $\lambda_{\text{BC}}$         | 1.0                           | Puntos iniciales               | 100                            |
| $\lambda_{\text{norm}}$       | 3.0                           | Añadidos por refinamiento      | 50                             |
| $\lambda_{\text{sym}}$        | 5.0                           | Puntos evaluación              | 500                            |
| $\lambda_{\text{proj}}$       | 0.0 ($n=0$), 10.0 ($n>0$)   | Frecuencia adaptación          | cada 1000 épocas               |
| Semilla aleatoria             | 42                            | Criterio de adición            | mayor residuo absoluto         |
| Optimizador                   | Adam                          | LR inicial                      | $5 \times 10^{-5}$             |
| LR tras puntos máximos        | $1 \times 10^{-4}$            | Puntos máximos                 | 1000                           |
| Criterio de parada            | estabilización energética     |                               |                               |

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

## Adaptive Mesh Training

In the adaptive mesh approach, training begins with a relatively small initial mesh and a high learning rate, which allows the neural network and the refinement algorithm to progressively "build" the mesh. During the first stage, points are added based on a residual-based criterion: at every fixed number of epochs, the algorithm identifies regions where the PDE residual is largest and adds new points there, increasing the spatial resolution adaptively. Once the maximum number of points allowed in the mesh is reached, the second stage begins, where the learning rate is reduced to favor fine convergence while keeping the mesh fixed at its optimized configuration. This strategy ensures that the network focuses computational resources on regions where the solution is more complex or the residual is higher, improving accuracy while avoiding unnecessary evaluations in smooth regions.  



### Results for Adaptive Mesh

![Wave function n=0 adaptive](psi_n0_adap.png)  
![Wave function n=1 adaptive](psi_n1_adap.png)

---

## Fixed Mesh Training

For the fixed mesh approach, the total number of points, convergence criteria, and hyperparameters are set by reference to the adaptive mesh results, providing a fair comparison under equivalent conditions of resolution and training. In this case, the mesh is uniform and fixed throughout the entire training process. The learning rate is kept constant or adjusted according to the fine-tuning stage observed in the adaptive method. Unlike the adaptive strategy, no additional points are added; the network must learn the solution using the pre-defined mesh. This fixed mesh method serves as a baseline to evaluate the benefits of adaptively refining the mesh based on the PDE residual.


### Results for Fixed Mesh

![Wave function n=0 fixed](psi_n0_fija.png)  
![Wave function n=1 fixed](psi_n1_fija.png)
