"""
Exact Analytical Solution for Fisher-KPP Equation

This module computes the exact traveling wave solution:
u(x,t) = 1 / (1 + exp(√(R/2D) * (x - √(2DR) * t)))

Based on the paper by Ahmed Aberqi and Ahmed Miloudi (arXiv:2601.11406v1)
"""

import numpy as np
import matplotlib.pyplot as plt


class ExactSolution:
    """Exact analytical solution for Fisher-KPP equation"""
    
    def __init__(self, D=0.01, R=1.0):
        """
        Initialize with physical parameters
        
        Parameters:
        -----------
        D : float
            Diffusion coefficient (default: 0.01)
        R : float
            Reaction rate (default: 1.0)
        """
        self.D = D
        self.R = R
        
        # Pre-compute constants
        self.sqrt_term = np.sqrt(R / (2 * D))
        self.wave_speed = np.sqrt(2 * D * R)
        
        print(f"Exact Solution Parameters:")
        print(f"  D = {D}")
        print(f"  R = {R}")
        print(f"  Wave speed c = √(2DR) = {self.wave_speed:.6f}")
        print(f"  √(R/2D) = {self.sqrt_term:.6f}")
    
    def compute(self, x, t):
        """
        Compute exact solution at given points
        
        u(x,t) = 1 / (1 + exp(√(R/2D) * (x - √(2DR) * t)))
        
        Parameters:
        -----------
        x : array-like
            Spatial coordinates
        t : float or array-like
            Time coordinate(s)
        
        Returns:
        --------
        u : array
            Exact solution values
        """
        exponent = self.sqrt_term * (x - self.wave_speed * t)
        u_exact = 1.0 / (1.0 + np.exp(exponent))
        return u_exact
    
    def compute_from_data_samples(self, filepath):
        """
        Load data samples and compute exact solution at those points
        
        Parameters:
        -----------
        filepath : str
            Path to .npz file containing sample points
            Expected format: arrays with columns [x, t] (note: x first!)
        
        Returns:
        --------
        results : dict
            Dictionary containing exact solutions at each sample type
        """
        # Load data
        data = np.load(filepath)
        
        print(f"\nLoading samples from: {filepath}")
        print(f"Available sample types: {data.files}")
        print(f"\nNote: Dataset uses (x,t) ordering")
        
        results = {}
        
        # Process each sample type
        for key in data.files:
            points = data[key]
            print(f"\n{key}: {points.shape[0]} points")
            
            # Extract x and t coordinates (dataset is in x,t order)
            x_coords = points[:, 0]
            t_coords = points[:, 1]
            
            # Compute exact solution
            u_exact = self.compute(x_coords, t_coords)
            
            # Store results
            results[key] = {
                'points': points,
                'x': x_coords,
                't': t_coords,
                'u_exact': u_exact
            }
            
            print(f"  Exact solution computed")
            print(f"    Mean: {np.mean(u_exact):.6f}")
            print(f"    Min:  {np.min(u_exact):.6f}")
            print(f"    Max:  {np.max(u_exact):.6f}")
        
        return results
    
    def save_results(self, results, output_path):
        """
        Save computed exact solutions to file
        
        Parameters:
        -----------
        results : dict
            Results from compute_from_data_samples
        output_path : str
            Path to save .npz file
        """
        save_dict = {}
        
        for key, data in results.items():
            save_dict[f'{key}_points'] = data['points']
            save_dict[f'{key}_t'] = data['t']
            save_dict[f'{key}_x'] = data['x']
            save_dict[f'{key}_u_exact'] = data['u_exact']
        
        np.savez(output_path, **save_dict)
        print(f"\nResults saved to: {output_path}")
    
    def plot_results(self, results, save_path=None):
        """
        Visualize exact solution at sample points
        
        Parameters:
        -----------
        results : dict
            Results from compute_from_data_samples
        save_path : str, optional
            Path to save figure
        """
        n_types = len(results)
        fig, axes = plt.subplots(1, n_types, figsize=(5*n_types, 4))
        
        if n_types == 1:
            axes = [axes]
        
        for idx, (key, data) in enumerate(results.items()):
            ax = axes[idx]
            
            scatter = ax.scatter(data['x'], data['t'], 
                               c=data['u_exact'], 
                               cmap='viridis', 
                               s=10, 
                               alpha=0.6)
            
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('t', fontsize=12)
            ax.set_title(f'{key.capitalize()}\n({len(data["x"])} points)', fontsize=12)
            
            plt.colorbar(scatter, ax=ax, label='u(x,t)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig


def main():
    """Main function demonstrating exact solution computation"""
    
    print("="*70)
    print("Fisher-KPP Exact Analytical Solution Calculator")
    print("="*70)
    
    # Initialize with paper parameters
    exact_sol = ExactSolution(D=0.01, R=1.0)
    
    # Load data samples and compute exact solution
    results = exact_sol.compute_from_data_samples('data_samples.npz')
    
    # Save results
    exact_sol.save_results(results, 'exact_solution_results.npz')
    
    # Visualize
    exact_sol.plot_results(results, save_path='exact_solution_plot.png')
    
    print("\n" + "="*70)
    print("Computation complete!")
    print("="*70)
    
    return exact_sol, results


if __name__ == "__main__":
    exact_sol, results = main()
    plt.show()