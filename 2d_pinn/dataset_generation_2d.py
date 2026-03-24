import numpy as np
import os

def generate_2d_data(
    num_initial=1000,
    num_boundary=4000,
    num_collocation=10000,
    output_path="data_samples_2d.npz",
):
    """
    Generates training data for a 2D Fisher-KPP problem.
    Domain: x in [0, 1], y in [0, 1], t in [0, 1]

    Args:
        num_initial (int): Number of initial condition points (t=0).
        num_boundary (int): Number of boundary condition points (split across 4 boundaries).
        num_collocation (int): Number of collocation points for the PDE residual.
        output_path (str): Path to save the .npz file.
    """
    print("Generating 2D dataset...")

    # 1. Initial Condition (t=0)
    # Randomly sample exactly num_initial points in the x-y plane at t=0.
    x_ic = np.random.rand(num_initial, 1)
    y_ic = np.random.rand(num_initial, 1)
    t_ic = np.zeros((num_initial, 1))
    initial_points = np.hstack((x_ic, y_ic, t_ic))

    # 2. Boundary Conditions (t>0 on x=0, x=1, y=0, y=1)
    num_per_boundary = num_boundary // 4
    t_bc = np.random.rand(num_per_boundary, 1)
    x_bc_vals = np.random.rand(num_per_boundary, 1)
    y_bc_vals = np.random.rand(num_per_boundary, 1)

    # Boundary x=0
    bc_x0 = np.hstack((np.zeros_like(t_bc), y_bc_vals, t_bc))
    # Boundary x=1
    bc_x1 = np.hstack((np.ones_like(t_bc), y_bc_vals, t_bc))
    # Boundary y=0
    bc_y0 = np.hstack((x_bc_vals, np.zeros_like(t_bc), t_bc))
    # Boundary y=1
    bc_y1 = np.hstack((x_bc_vals, np.ones_like(t_bc), t_bc))

    boundary_points = np.vstack([bc_x0, bc_x1, bc_y0, bc_y1])

    # 3. Collocation Points (randomly sampled in the full domain)
    x_col = np.random.rand(num_collocation, 1)
    y_col = np.random.rand(num_collocation, 1)
    t_col = np.random.rand(num_collocation, 1)
    collocation_points = np.hstack((x_col, y_col, t_col))

    print(f"  Initial points     : {initial_points.shape[0]:,}")
    print(f"  Boundary points    : {boundary_points.shape[0]:,}")
    print(f"  Collocation points : {collocation_points.shape[0]:,}")

    # Save to .npz file
    np.savez(
        output_path,
        initial=initial_points,
        boundary=boundary_points,
        collocation=collocation_points,
    )
    print(f"Dataset saved to '{os.path.abspath(output_path)}'")


if __name__ == "__main__":
    generate_2d_data()
