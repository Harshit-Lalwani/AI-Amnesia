import os
import numpy as np
import time
from tqdm import tqdm

def initial_condition_2d(x, y):
    """A Gaussian-like initial condition for the 2D problem."""
    return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

def solve_fisher_kpp_2d_fdm(
    D=0.01, R=1.0,
    Lx=1.0, Ly=1.0, T=1.0,
    Nx=51, Ny=51, Nt=10000
):
    """
    Solves the 2D Fisher-KPP equation using the Finite Difference Method
    (Forward Time, Central Space - FTCS).

    u_t = D * (u_xx + u_yy) + R * u * (1 - u)

    Args:
        D (float): Diffusion coefficient.
        R (float): Reaction rate.
        Lx (float): Domain length in x.
        Ly (float): Domain length in y.
        T (float): Total time to solve for.
        Nx (int): Number of spatial points in x.
        Ny (int): Number of spatial points in y.
        Nt (int): Number of time steps.

    Returns:
        tuple: (u_final, x, y, t) where u_final is the solution at time T.
    """
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / Nt

    # Stability check for the explicit method
    # The condition is dt <= 1 / (2*D * (1/dx^2 + 1/dy^2))
    # We add the reaction term for a more conservative estimate
    max_dt = 1.0 / (2.0 * D * (1.0/dx**2 + 1.0/dy**2) + R)
    if dt > max_dt:
        print(f"Warning: dt={dt:.6f} may be too large for stability.")
        print(f"         Recommended max_dt <= {max_dt:.6f}")
        # For this problem, we will proceed but this is a critical check.

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t = np.linspace(0, T, Nt + 1)

    # Initialize solution array u
    u = np.zeros((Nx, Ny, Nt + 1))

    # Set initial condition
    X, Y = np.meshgrid(x, y, indexing='ij')
    u[:, :, 0] = initial_condition_2d(X, Y)

    # Time-stepping loop
    for n in tqdm(range(Nt), desc="FDM Solver Progress"):
        # Enforce zero Dirichlet boundary conditions
        u[0, :, n] = 0
        u[-1, :, n] = 0
        u[:, 0, n] = 0
        u[:, -1, n] = 0

        # Update interior points
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                # Central differences for spatial derivatives
                u_xx = (u[i+1, j, n] - 2*u[i, j, n] + u[i-1, j, n]) / dx**2
                u_yy = (u[i, j+1, n] - 2*u[i, j, n] + u[i, j-1, n]) / dy**2

                # Reaction term
                reaction = R * u[i, j, n] * (1 - u[i, j, n])

                # Forward difference for time derivative
                u[i, j, n+1] = u[i, j, n] + dt * (D * (u_xx + u_yy) + reaction)

    # Enforce boundary conditions on the final state as well
    u[0, :, -1] = 0
    u[-1, :, -1] = 0
    u[:, 0, -1] = 0
    u[:, -1, -1] = 0

    return u[:, :, -1], x, y, t[-1]

if __name__ == '__main__':
    print("Starting 2D FDM solver for Fisher-KPP equation...")
    t0 = time.time()

    # Parameters matching the PINN setup
    # Using a higher Nt for stability with the explicit method.
    # A finer time discretization is needed compared to what a PINN might handle implicitly.
    u_final, x_grid, y_grid, final_time = solve_fisher_kpp_2d_fdm(
        D=0.01, R=1.0,
        Lx=1.0, Ly=1.0, T=1.0,
        Nx=51, Ny=51, Nt=20000
    )

    elapsed = time.time() - t0
    print(f"\nFDM simulation completed in {elapsed:.2f} seconds.")

    # The PINN evaluates at t=1.0, so we save the grid and the final slice.
    # The PINN test points are flattened, so we flatten the FDM solution to match.
    u_flat = u_final.T.flatten().reshape(-1, 1) # Transpose to match meshgrid 'xy' vs 'ij'

    # Create meshgrid for saving, matching the 'xy' convention used in PINN notebook
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    x_flat = X_mesh.flatten().reshape(-1, 1)
    y_flat = Y_mesh.flatten().reshape(-1, 1)
    t_flat = np.full_like(x_flat, final_time)

    save_path = 'results/2d/fdm_solution_2d.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        u_exact=u_flat,
        x=x_flat,
        y=y_flat,
        t=t_flat
    )
    print(f"FDM solution saved to '{save_path}'")
    print(f"  Solution shape: {u_flat.shape}")
    print(f"  Grid shape (x, y, t): {x_flat.shape}")
