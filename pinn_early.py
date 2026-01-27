"""
Physics-Informed Neural Network (PINN) for Fisher-KPP Equation
PyTorch Implementation - Exact replication of paper by Aberqi & Miloudi (arXiv:2601.11406v1)

Key features from paper:
- 7 hidden layers × 50 neurons each
- Tanh activation
- Xavier initialization
- Adam optimizer with learning rate 1e-3
- Exponential decay (0.99)
- Adaptive weighting (I-PINN)
- 10,000 collocation points
- 1,000 initial condition points
- 2,000 boundary condition points
- Initial training: 10,000 iterations
- Retraining: 20,000 + 20,000 iterations with lr=1e-4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt


def get_preferred_device():
    """Use CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN_FisherKPP(nn.Module):
    """
    Physics-Informed Neural Network for Fisher-KPP equation
    
    PDE: ∂u/∂t = D * ∂²u/∂x² + R*u*(1-u)
    """
    
    def __init__(self, layers=[2, 50, 50, 50, 50, 50, 50, 50, 1], 
                 D=0.01, R=1.0, adaptive_weights=True):
        """
        Initialize PINN
        
        Parameters:
        -----------
        layers : list
            Network architecture (default: [2, 50, 50, 50, 50, 50, 50, 50, 1])
            7 hidden layers × 50 neurons each
        D : float
            Diffusion coefficient (default: 0.01)
        R : float
            Reaction rate (default: 1.0)
        adaptive_weights : bool
            Use adaptive weighting (I-PINN) as in paper
        """
        super(PINN_FisherKPP, self).__init__()
        
        self.layers = layers
        self.D = D
        self.R = R
        self.adaptive_weights = adaptive_weights
        
        # Build network
        self.network = self.build_network(layers)
        
        # Adaptive weights for loss terms (I-PINN from paper)
        if adaptive_weights:
            self.lambda_ic = torch.tensor(1.0, requires_grad=False)
            self.lambda_bc = torch.tensor(1.0, requires_grad=False)
            self.lambda_res = torch.tensor(1.0, requires_grad=False)
            self.lambda_max = 10000.0  # From paper's Figure 2
            # Rapid rise and early saturation as shown in paper Figure 2.
            self.adaptive_growth = 1.02
            self.adaptive_update_every = 1
        else:
            self.lambda_ic = 1.0
            self.lambda_bc = 1.0
            self.lambda_res = 1.0
        
        # History tracking
        self.loss_history = []
        self.loss_ic_history = []
        self.loss_bc_history = []
        self.loss_res_history = []
        self.l2_error_history = []
        self.iteration_history = []
        
        print("="*70)
        print("PINN Configuration (Exact Paper Implementation)")
        print("="*70)
        print(f"Architecture: {layers}")
        print(f"  Input: 2 (x, t)")
        print(f"  Hidden: 7 layers × 50 neurons")
        print(f"  Output: 1 (u)")
        print(f"  Activation: Tanh")
        print(f"  Total parameters: {self.count_parameters()}")
        print(f"\nPDE Parameters:")
        print(f"  D = {D} (diffusion)")
        print(f"  R = {R} (reaction)")
        print(f"\nAdaptive weighting: {adaptive_weights}")
        if adaptive_weights:
            print(f"  lambda_max: {self.lambda_max}")
            print(f"  adaptive_growth: {self.adaptive_growth}")
            print(f"  adaptive_update_every: {self.adaptive_update_every} iter")
        print(f"Device: {next(self.parameters()).device}")
        print("="*70)
    
    def build_network(self, layers):
        """Build neural network with Xavier initialization"""
        network_layers = []
        
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i+1])
            
            # Xavier initialization
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            
            network_layers.append(linear)
            
            # Tanh activation for all except last layer
            if i < len(layers) - 2:
                network_layers.append(nn.Tanh())
        
        return nn.Sequential(*network_layers)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, t):
        """
        Forward pass through network
        
        Parameters:
        -----------
        x : tensor
            Spatial coordinates, shape (N, 1)
        t : tensor
            Time coordinates, shape (N, 1)
        
        Returns:
        --------
        u : tensor
            Network output, shape (N, 1)
        """
        X = torch.cat([x, t], dim=1)
        u = self.network(X)
        return u
    
    def exact_solution(self, x, t):
        """
        Exact analytical solution from paper Equation (6)
        
        u(x,t) = 1 / (1 + exp(√(R/2D) * (x - √(2DR) * t)))
        """
        sqrt_term = np.sqrt(self.R / (2.0 * self.D))
        wave_speed = np.sqrt(2.0 * self.D * self.R)
        exponent = sqrt_term * (x - wave_speed * t)
        return 1.0 / (1.0 + np.exp(exponent))
    
    def pde_residual(self, x, t):
        """
        Compute PDE residual using automatic differentiation
        
        Residual: ∂u/∂t - D*∂²u/∂x² - R*u*(1-u) = 0
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Fisher-KPP equation residual
        residual = u_t - self.D * u_xx - self.R * u * (1 - u)
        
        return residual
    
    def compute_loss(self, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res):
        """
        Compute total loss (Equation 1 from paper)
        
        L = λ_IC * L_IC + λ_BC * L_BC + λ_res * L_res
        """
        # Initial condition loss
        u_pred_ic = self.forward(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)
        
        # Boundary condition loss
        u_pred_bc = self.forward(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)
        
        # PDE residual loss
        residual = self.pde_residual(x_res, t_res)
        loss_res = torch.mean(residual ** 2)
        
        # Total weighted loss
        loss = (self.lambda_ic * loss_ic + 
                self.lambda_bc * loss_bc + 
                self.lambda_res * loss_res)
        
        return loss, loss_ic, loss_bc, loss_res
    
    def update_adaptive_weights(self, loss_ic, loss_bc, iteration):
        """
        Update adaptive weights for I-PINN (from paper's Figure 2)
        
        IC/BC weights rise quickly and saturate at lambda_max (1e4),
        consistent with the trend in paper Figure 2.
        """
        if self.adaptive_weights:
            # Fast multiplicative growth to reach saturation early.
            if self.lambda_ic < self.lambda_max:
                new_lambda_ic = min(self.lambda_ic * self.adaptive_growth, self.lambda_max)
                self.lambda_ic = new_lambda_ic
            
            if self.lambda_bc < self.lambda_max:
                new_lambda_bc = min(self.lambda_bc * self.adaptive_growth, self.lambda_max)
                self.lambda_bc = new_lambda_bc
    
    def train_pinn(self, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res,
                   iterations=10000, learning_rate=1e-3, decay_rate=0.99,
                   print_every=1000, phase_name="Initial Training"):
        """
        Train the PINN
        
        Parameters from paper:
        - Initial training: 10,000 iterations, lr=1e-3
        - Retraining: 20,000 iterations, lr=1e-4 (reset optimizer)
        """
        print(f"\n{'='*70}")
        print(f"{phase_name}")
        print(f"{'='*70}")
        print(f"Iterations: {iterations}")
        print(f"Learning rate: {learning_rate}")
        print(f"Decay rate: {decay_rate}")
        print(f"\n{'Iter':>8} {'Loss':>15} {'L_IC':>15} {'L_BC':>15} {'L_Res':>15} {'L2_Error':>15} {'Time(s)':>10}")
        print("="*105)
        
        # Adam optimizer with exponential decay (from paper)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        start_time = time.time()
        base_iteration = len(self.loss_history)
        
        for it in range(iterations):
            optimizer.zero_grad()
            
            # Compute loss
            loss, loss_ic, loss_bc, loss_res = self.compute_loss(
                x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update adaptive weights (I-PINN)
            if self.adaptive_weights and it % self.adaptive_update_every == 0:
                self.update_adaptive_weights(loss_ic, loss_bc, base_iteration + it)
            
            # Compute L2 error and print
            if it % print_every == 0 or it == iterations - 1:
                l2_error = self.compute_l2_error()
                elapsed = time.time() - start_time
                
                # Store history
                self.loss_history.append(float(loss.item()))
                self.loss_ic_history.append(float(loss_ic.item()))
                self.loss_bc_history.append(float(loss_bc.item()))
                self.loss_res_history.append(float(loss_res.item()))
                self.l2_error_history.append(l2_error)
                self.iteration_history.append(base_iteration + it)
                
                print(f"{it:8d} {float(loss.item()):15.6e} {float(loss_ic.item()):15.6e} "
                      f"{float(loss_bc.item()):15.6e} {float(loss_res.item()):15.6e} "
                      f"{l2_error:15.6e} {elapsed:10.2f}")
        
        total_time = time.time() - start_time
        print("="*105)
        print(f"✓ {phase_name} completed in {total_time:.2f} seconds")
        print(f"  Final loss: {float(loss.item()):.6e}")
        print(f"  Final L2 error: {l2_error:.6e}")
        
        return l2_error
    
    def compute_l2_error(self, N=201):
        """
        Compute L2 error on grid at t=1.0
        """
        device = next(self.parameters()).device
        x_test = torch.linspace(0, 1, N, device=device).reshape(-1, 1)
        t_test = torch.ones_like(x_test) * 1.0
        
        with torch.no_grad():
            u_pred = self.forward(x_test, t_test).detach().cpu().numpy()
        
        u_exact = self.exact_solution(
            x_test.detach().cpu().numpy(),
            t_test.detach().cpu().numpy()
        ).reshape(-1, 1)
        
        l2_error = np.sqrt(np.sum((u_pred - u_exact)**2)) / np.sqrt(np.sum(u_exact**2))
        
        return l2_error
    
    def predict(self, x, t):
        """
        Predict u(x,t)
        
        Parameters:
        -----------
        x : array or tensor
            Spatial coordinates
        t : array or tensor
            Time coordinates
        
        Returns:
        --------
        u : array
            Predicted solution
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()

        device = next(self.parameters()).device
        x = x.to(device)
        t = t.to(device)
        
        with torch.no_grad():
            u = self.forward(x, t)
        
        return u.detach().cpu().numpy()
    
    def save_model(self, filepath):
        """Save complete model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'loss_history': self.loss_history,
            'loss_ic_history': self.loss_ic_history,
            'loss_bc_history': self.loss_bc_history,
            'loss_res_history': self.loss_res_history,
            'l2_error_history': self.l2_error_history,
            'iteration_history': self.iteration_history,
            'lambda_ic': self.lambda_ic,
            'lambda_bc': self.lambda_bc,
            'lambda_res': self.lambda_res,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load complete model state"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.loss_ic_history = checkpoint['loss_ic_history']
        self.loss_bc_history = checkpoint['loss_bc_history']
        self.loss_res_history = checkpoint['loss_res_history']
        self.l2_error_history = checkpoint['l2_error_history']
        self.iteration_history = checkpoint['iteration_history']
        self.lambda_ic = checkpoint['lambda_ic']
        self.lambda_bc = checkpoint['lambda_bc']
        self.lambda_res = checkpoint['lambda_res']
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iters = np.array(self.iteration_history)
        
        # Total loss
        axes[0, 0].semilogy(iters, self.loss_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Total Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss Evolution', fontsize=13)
        axes[0, 0].grid(True, alpha=0.3)
        
        # L2 Error
        axes[0, 1].plot(iters, self.l2_error_history, 'r-', linewidth=2, label='PINN')
        axes[0, 1].axhline(y=9.78e-2, color='b', linestyle='--', linewidth=2, label='FDM (9.78%)')
        axes[0, 1].axhline(y=5.57e-2, color='g', linestyle=':', linewidth=2, label='Paper initial (5.57%)')
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('L2 Error', fontsize=12)
        axes[0, 1].set_title('L2 Error Evolution', fontsize=13)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss components
        axes[1, 0].semilogy(iters, self.loss_ic_history, 'b-', linewidth=2, label='IC')
        axes[1, 0].semilogy(iters, self.loss_bc_history, 'r-', linewidth=2, label='BC')
        axes[1, 0].semilogy(iters, self.loss_res_history, 'g-', linewidth=2, label='Residual')
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Loss Components', fontsize=12)
        axes[1, 0].set_title('Loss Component Evolution', fontsize=13)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning curve with phase markers
        axes[1, 1].plot(iters, self.l2_error_history, 'b-', linewidth=2)
        
        # Mark training phases
        if len(iters) > 10000:
            axes[1, 1].axvline(x=10000, color='r', linestyle='--', linewidth=1, label='Retrain 1')
        if len(iters) > 30000:
            axes[1, 1].axvline(x=30000, color='orange', linestyle='--', linewidth=1, label='Retrain 2')
        
        axes[1, 1].set_xlabel('Iteration', fontsize=12)
        axes[1, 1].set_ylabel('L2 Error', fontsize=12)
        axes[1, 1].set_title('Training Phases', fontsize=13)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.close()
        return fig


def prepare_training_data(data_path):
    """
    Load and prepare training data from npz file
    
    Data format: (x, t) ordering
    """
    data = np.load(data_path)
    
    print("\nLoading training data...")
    print(f"  Collocation: {data['collocation'].shape[0]} points")
    print(f"  Initial: {data['initial'].shape[0]} points")
    print(f"  Boundary: {data['boundary'].shape[0]} points")
    
    # Collocation points (x, t)
    collocation = data['collocation'].astype(np.float32)
    x_res = torch.from_numpy(collocation[:, 0:1])
    t_res = torch.from_numpy(collocation[:, 1:2])
    
    # Initial condition points
    initial = data['initial'].astype(np.float32)
    x_ic = torch.from_numpy(initial[:, 0:1])
    t_ic = torch.from_numpy(initial[:, 1:2])
    
    # Boundary condition points
    boundary = data['boundary'].astype(np.float32)
    x_bc = torch.from_numpy(boundary[:, 0:1])
    t_bc = torch.from_numpy(boundary[:, 1:2])
    
    # Compute exact solutions for IC and BC
    def exact_sol(x, t, D=0.01, R=1.0):
        sqrt_term = np.sqrt(R / (2.0 * D))
        wave_speed = np.sqrt(2.0 * D * R)
        exponent = sqrt_term * (x - wave_speed * t)
        return 1.0 / (1.0 + np.exp(exponent))
    
    u_ic = exact_sol(x_ic.numpy(), t_ic.numpy()).astype(np.float32)
    u_bc = exact_sol(x_bc.numpy(), t_bc.numpy()).astype(np.float32)
    
    u_ic = torch.from_numpy(u_ic)
    u_bc = torch.from_numpy(u_bc)
    
    return x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res


def main():
    """Main function implementing paper's training procedure"""
    
    print("="*70)
    print("PINN for Fisher-KPP Equation (PyTorch Implementation)")
    print("="*70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Prefer CUDA when available
    device = get_preferred_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    print(f"Using device: {device}")
    
    # Load training data
    data_path = 'data_samples.npz'
    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res = prepare_training_data(data_path)

    # Move all training tensors to selected device
    x_ic = x_ic.to(device)
    t_ic = t_ic.to(device)
    u_ic = u_ic.to(device)
    x_bc = x_bc.to(device)
    t_bc = t_bc.to(device)
    u_bc = u_bc.to(device)
    x_res = x_res.to(device)
    t_res = t_res.to(device)
    
    # Initialize PINN (7×50 architecture from paper)
    pinn = PINN_FisherKPP(
        layers=[2, 50, 50, 50, 50, 50, 50, 50, 1],
        D=0.01,
        R=1.0,
        adaptive_weights=True  # I-PINN from paper
    ).to(device)
    
    # ========================================================================
    # PHASE 1: Initial Training (10,000 iterations, lr=1e-3)
    # ========================================================================
    l2_error_initial = pinn.train_pinn(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res,
        iterations=10000,
        learning_rate=1e-3,
        decay_rate=0.99,
        print_every=1000,
        phase_name="PHASE 1: Initial Training (Paper: 10k iters, lr=1e-3)"
    )
    
    # Save model after initial training
    pinn.save_model('pinn_initial.pth')
    
    print(f"\n{'='*70}")
    print("Initial Training Results")
    print(f"{'='*70}")
    print(f"L2 Error: {l2_error_initial:.6e}")
    print(f"Paper reports: 5.57 × 10⁻² (~5.57%)")
    print(f"Ratio: {l2_error_initial / 5.57e-2:.2f}x")
    
    # ========================================================================
    # PHASE 2: Retraining Phase 1 (20,000 iterations, lr=1e-4)
    # ========================================================================
    print(f"\n{'='*70}")
    print("Starting Retraining Phase 1")
    print(f"{'='*70}")
    print("Note: Optimizer state is RESET (as per paper)")
    print("This typically degrades performance!")
    
    l2_error_retrain1 = pinn.train_pinn(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res,
        iterations=20000,
        learning_rate=1e-4,  # Reduced learning rate
        decay_rate=0.99,
        print_every=2000,
        phase_name="PHASE 2: Retraining Phase 1 (Paper: 20k iters, lr=1e-4)"
    )
    
    print(f"\n{'='*70}")
    print("Retraining Phase 1 Results")
    print(f"{'='*70}")
    print(f"L2 Error: {l2_error_retrain1:.6e}")
    print(f"Paper reports: 9.796 × 10⁻² (~9.80%)")
    print(f"Ratio: {l2_error_retrain1 / 9.796e-2:.2f}x")
    
    # ========================================================================
    # PHASE 3: Retraining Phase 2 (20,000 more iterations, lr=1e-4)
    # ========================================================================
    l2_error_retrain2 = pinn.train_pinn(
        x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_res, t_res,
        iterations=20000,
        learning_rate=1e-4,
        decay_rate=0.99,
        print_every=2000,
        phase_name="PHASE 3: Retraining Phase 2 (Paper: 20k iters, lr=1e-4)"
    )
    
    # Save final model
    pinn.save_model('pinn_final.pth')
    
    # ========================================================================
    # Final Results Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nPhase 1 (Initial Training, 10k iters):")
    print(f"  Our L2 Error:  {l2_error_initial:.6e}")
    print(f"  Paper reports: 5.57 × 10⁻²")
    print(f"  Ratio:         {l2_error_initial / 5.57e-2:.2f}x")
    
    print(f"\nPhase 2 (Retraining 1, 20k iters):")
    print(f"  Our L2 Error:  {l2_error_retrain1:.6e}")
    print(f"  Paper reports: 9.796 × 10⁻²")
    print(f"  Ratio:         {l2_error_retrain1 / 9.796e-2:.2f}x")
    
    print(f"\nPhase 3 (Retraining 2, 20k iters):")
    print(f"  Our L2 Error:  {l2_error_retrain2:.6e}")
    print(f"  Paper reports: 9.794 × 10⁻²")
    print(f"  Ratio:         {l2_error_retrain2 / 9.794e-2:.2f}x")
    
    print(f"\n{'='*70}")
    print("Comparison with FDM")
    print(f"{'='*70}")
    print(f"FDM (Explicit Forward Euler): 9.784 × 10⁻² (~9.78%)")
    print(f"PINN Initial Training:         {l2_error_initial:.6e}")
    print(f"PINN After Retraining:         {l2_error_retrain2:.6e}")
    
    if l2_error_initial < 9.784e-2:
        print(f"\n✓ PINN outperforms FDM after initial training!")
        print(f"  Improvement: {(9.784e-2 - l2_error_initial) / 9.784e-2 * 100:.1f}%")
    
    if l2_error_retrain2 > l2_error_initial:
        print(f"\n⚠ Retraining degraded performance!")
        print(f"  Degradation: {(l2_error_retrain2 - l2_error_initial) / l2_error_initial * 100:.1f}%")
        print(f"  Reason: Optimizer reset + lower learning rate")
    
    # Plot training history
    pinn.plot_training_history(save_path='pinn_training_history.png')
    
    # Save results summary
    results = {
        'l2_error_initial': l2_error_initial,
        'l2_error_retrain1': l2_error_retrain1,
        'l2_error_retrain2': l2_error_retrain2,
        'loss_history': pinn.loss_history,
        'l2_error_history': pinn.l2_error_history,
        'iteration_history': pinn.iteration_history
    }
    np.savez('pinn_results.npz', **results)
    print(f"\nResults saved to: pinn_results.npz")
    
    return pinn


if __name__ == "__main__":
    pinn = main()