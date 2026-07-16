import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def get_preferred_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_preferred_device()
print(f"Using device: {device}")


class PINN_FisherKPP(nn.Module):
    """
    7×50 Physics-Informed Neural Network for Fisher-KPP equation.

    Architecture (paper Section 3.3, Figure 1):
        Input  : (x, t)  — 2 neurons          [FIX-1: consistent (x,t) order]
        Hidden : 7 layers × 50 neurons, Tanh
        Output : u(x, t) — 1 neuron

    Adaptive weighting (I-PINN, paper Figure 2):
        lambda_ic, lambda_bc  — monotone non-decreasing, cap at lambda_max
        lambda_res            — fixed at 1.0 throughout
    """

    def __init__(self,
                 layers=(2, 50, 50, 50, 50, 50, 50, 50, 1),
                 D=0.01, R=1.0,
                 adaptive_weights=True):
        super().__init__()
        self.D = D
        self.R = R
        self.adaptive_weights = adaptive_weights

        # FIX-3 + FIX-4: plain float lambdas; lambda_max=100 keeps L_Res
        # in the gradient mix (10,000 drowns it out 10,000:1)
        self.lambda_ic  = 1.0   # FIX-4: float, not torch.tensor
        self.lambda_bc  = 1.0
        self.lambda_res = 1.0   # paper: fixed at 1.0 always
        self.lambda_max = 10000.0 # Reverted from 100.0 to 10000.0 to match paper's Figure 2b

        self.network = self._build_network(layers)

        # Training history
        self.loss_history      = []
        self.loss_ic_history   = []
        self.loss_bc_history   = []
        self.loss_res_history  = []
        self.l2_error_history  = []
        self.lambda_ic_history = []
        self.lambda_bc_history = []
        self.iteration_history = []

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 70)
        print("PINN Configuration (Paper: Aberqi & Miloudi, arXiv:2601.11406v1)")
        print("=" * 70)
        print(f"  Architecture : {list(layers)}")
        print(f"  Activation   : Tanh  (paper Section 3.3)")
        print(f"  Init         : Xavier normal weights, zero biases")
        print(f"  Parameters   : {n_params:,}")
        print(f"  D={D}, R={R}")
        print(f"  lambda_max   : {self.lambda_max}  (IC/BC cap)")
        print(f"  Device       : {next(self.parameters()).device}")
        print("=" * 70)

    # ----------------------------------------------------------
    def _build_network(self, layers):
        """Fully-connected net with Xavier init (paper Section 3.3)."""
        mods = []
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight)   # paper Section 3.3
            nn.init.zeros_(lin.bias)              # paper Section 3.3
            mods.append(lin)
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        return nn.Sequential(*mods)

    # ----------------------------------------------------------
    def forward(self, x, t):
        """
        Forward pass.

        FIX-1: signature is forward(x, t) — data loader gives (x, t) order.
        Paper Figure 1 shows (t, x) as diagram labels only; what matters is
        that the concatenation order is consistent with how data is loaded.
        The dataset stores columns as [x, t], so x is always the first
        positional argument here.
        """
        return self.network(torch.cat([x, t], dim=1))

    # ----------------------------------------------------------
    def exact_solution(self, x, t):
        """Exact analytical solution (paper Equation 6)."""
        sqrt_term  = np.sqrt(self.R / (2.0 * self.D))
        wave_speed = np.sqrt(2.0 * self.D * self.R)
        exponent   = sqrt_term * (x - wave_speed * t)
        return 1.0 / (1.0 + np.exp(exponent))

    # ----------------------------------------------------------
    def pde_residual(self, x_r, t_r):
        """
        Fisher-KPP PDE residual via automatic differentiation (paper Eq. 5).

        Residual: du/dt - D*d²u/dx² - R*u*(1-u) = 0

        FIX-7: requires_grad is set on the tensors BEFORE calling this method
               (in train_pinn via .clone().detach().requires_grad_(True)).
               We do NOT mutate them here — that would corrupt the graph.
        """
        u = self.forward(x_r, t_r)

        u_t = torch.autograd.grad(
            u, t_r,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True)[0]

        u_x = torch.autograd.grad(
            u, x_r,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True)[0]

        u_xx = torch.autograd.grad(
            u_x, x_r,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True)[0]

        return u_t - self.D * u_xx - self.R * u * (1.0 - u)

    # ----------------------------------------------------------
    def compute_loss(self, x_ic, t_ic, u_ic,
                     x_bc, t_bc, u_bc,
                     x_r,  t_r):
        """
        Composite PINN loss (paper Equation 1):
            L = lambda_ic*L_IC + lambda_bc*L_BC + lambda_res*L_Res
        """
        loss_ic  = torch.mean((self.forward(x_ic, t_ic) - u_ic) ** 2)   # Eq.2
        loss_bc  = torch.mean((self.forward(x_bc, t_bc) - u_bc) ** 2)   # Eq.3
        res      = self.pde_residual(x_r, t_r)
        loss_res = torch.mean(res ** 2)                                   # Eq.4

        loss = (self.lambda_ic  * loss_ic +
                self.lambda_bc  * loss_bc +
                self.lambda_res * loss_res)
        return loss, loss_ic, loss_bc, loss_res

    # ----------------------------------------------------------
    def _update_adaptive_weights(self, loss_ic, loss_bc, loss_res):
        """
        I-PINN adaptive weight update (paper Figure 2, Wang et al. 2021).

        Target (Figure 2b):
          lambda_IC and lambda_BC rise steeply to lambda_max within ~1,000
          iterations then flatline permanently. lambda_res stays at 1.0.

        FIX-3 — three sub-fixes applied here:
          (a) Loss-ratio rule:  ratio = loss_res / loss_ic
              When IC is well-satisfied (loss_ic << loss_res), ratio is large
              → lambda jumps quickly to lambda_max. Matches Fig 2b saturation.

          (b) Monotone guard (max):
              lambda_ic = min(max(lambda_ic, ratio), lambda_max)
              Without max(), any transient increase in loss_ic pulls the ratio
              down and lambda DECREASES — causing the oscillation seen in
              the original graph. max() makes weights non-decreasing.

          (c) lambda_max = 100 (set in __init__):
              With 10,000 the IC/BC terms dominate the gradient 10,000:1,
              making L_Res contribute ~0 gradient — it flatlines immediately.
        """
        eps = 1e-10
        l_ic  = loss_ic.item()
        l_bc  = loss_bc.item()
        l_res = loss_res.item()

        ratio_ic = l_res / (l_ic + eps)
        ratio_bc = l_res / (l_bc + eps)

        # Monotone non-decreasing update (FIX-3b)
        self.lambda_ic = min(max(self.lambda_ic, ratio_ic), self.lambda_max)
        self.lambda_bc = min(max(self.lambda_bc, ratio_bc), self.lambda_max)
        # lambda_res intentionally NOT updated — stays at 1.0 (paper Fig. 2)

    # ----------------------------------------------------------
    @torch.no_grad()
    def compute_l2_error(self, x_test, t_test, u_exact_np):
        """
        Relative L2 error (paper Equation 8).

        FIX-5: accepts pre-built tensors to avoid per-call re-allocation.
        """
        u_pred = self.forward(x_test, t_test).cpu().numpy()
        num    = np.sqrt(np.sum((u_pred - u_exact_np) ** 2))
        denom  = np.sqrt(np.sum(u_exact_np ** 2))
        return float(num / denom)

    # ----------------------------------------------------------
    def train_pinn(self,
                   x_ic, t_ic, u_ic,
                   x_bc, t_bc, u_bc,
                   x_r,  t_r,
                   iterations=10_000,
                   learning_rate=1e-3,
                   decay_rate=0.99,
                   scheduler_step_every=100,
                   weight_warmup_iters=500,
                   print_every=1000,
                   phase_name="Training"):
        """
        Train for `iterations` steps with Adam + exponential LR decay.

        Key design decisions matching the paper:

        FIX-2  — scheduler.step() every `scheduler_step_every` iters (not every
                  step). Per-step stepping with gamma=0.99 kills LR to ~0 by
                  iter 1,000. Every-100-iters gives sensible final LR ~3.7×10⁻⁴.

        FIX-7  — collocation tensors cloned with requires_grad=True HERE,
                  once per phase, not inside pde_residual().

        FIX-5  — test tensors for L2 built once before the loop.

        FIX-3c — adaptive weight updates frozen for the first
                  `weight_warmup_iters` iterations to prevent abrupt jumps
                  at phase boundaries when loss ratios differ under new LR.

        Paper retraining protocol (Section 3.3):
                  A new optimizer is created each call → Adam state is reset
                  between phases (matches paper Table 1 "Adam (reset)").
        """
        # FIX-5: pre-build test grid once
        N_test     = 201
        x_t_np     = np.linspace(0, 1, N_test).reshape(-1, 1).astype(np.float32)
        t_t_np     = np.ones_like(x_t_np)
        u_exact_np = self.exact_solution(x_t_np, t_t_np).astype(np.float32)
        x_test     = torch.from_numpy(x_t_np).to(device)
        t_test     = torch.from_numpy(t_t_np).to(device)

        # FIX-7: set requires_grad once here, not inside pde_residual
        x_r = x_r.clone().detach().requires_grad_(True)
        t_r = t_r.clone().detach().requires_grad_(True)

        # Fresh Adam — optimizer state reset (paper Section 3.3)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # FIX-2: ExponentialLR stepped every scheduler_step_every iters
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        # Correctly calculate the global iteration offset for continuous plotting
        global_iteration_offset = self.iteration_history[-1] + 1 if self.iteration_history else 0
        t0   = time.time()

        header = (f"{'Iter':>8}  {'Loss':>13}  {'L_IC':>13}  {'L_BC':>13}  " \
                  f"{'L_Res':>13}  {'L2_Err':>12}  {'lam_IC':>8}  " \
                  f"{'lam_BC':>8}  {'LR':>10}  {'Time':>7}")
        sep = "=" * len(header)
        print(f"\n{sep}\n{phase_name}\n{sep}")
        print(f"  iters={iterations}  lr={learning_rate}  " \
              f"decay={decay_rate} per {scheduler_step_every} iters  " \
              f"warmup={weight_warmup_iters}")
        print(sep)
        print(header)
        print(sep)

        l2_err = self.compute_l2_error(x_test, t_test, u_exact_np)
        loss = loss_ic = loss_bc = loss_res = torch.tensor(0.0)

        for it in range(iterations):
            optimizer.zero_grad()
            loss, loss_ic, loss_bc, loss_res = self.compute_loss(
                x_ic, t_ic, u_ic,
                x_bc, t_bc, u_bc,
                x_r,  t_r)
            loss.backward()
            optimizer.step()

            # FIX-2: step scheduler every N iters, not every iter
            if (it + 1) % scheduler_step_every == 0:
                scheduler.step()

            # FIX-3c: freeze weight updates during warmup
            if (self.adaptive_weights
                    and (it + 1) % 100 == 0
                    and it >= weight_warmup_iters):
                self._update_adaptive_weights(loss_ic, loss_bc, loss_res)

            if it % print_every == 0 or it == iterations - 1:
                l2_err  = self.compute_l2_error(x_test, t_test, u_exact_np)
                cur_lr  = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0

                self.loss_history.append(loss.item())
                self.loss_ic_history.append(loss_ic.item())
                self.loss_bc_history.append(loss_bc.item())
                self.loss_res_history.append(loss_res.item())
                self.l2_error_history.append(l2_err)
                self.lambda_ic_history.append(self.lambda_ic)
                self.lambda_bc_history.append(self.lambda_bc)
                self.iteration_history.append(global_iteration_offset + it)

                print(f"{it:8d}  {loss.item():13.4e}  {loss_ic.item():13.4e}  " \
                      f"{loss_bc.item():13.4e}  {loss_res.item():13.4e}  " \
                      f"{l2_err:12.4e}  {self.lambda_ic:8.1f}  " \
                      f"{self.lambda_bc:8.1f}  {cur_lr:10.2e}  {elapsed:6.1f}s")

        elapsed_total = time.time() - t0
        print(sep)
        print(f"Done in {elapsed_total:.1f}s  |  " \
              f"loss={loss.item():.4e}  |  L2={l2_err:.4e}")
        return l2_err

    # ----------------------------------------------------------
    def save_model(self, filepath):
        torch.save({
            'model_state_dict':  self.state_dict(),
            'lambda_ic':         self.lambda_ic,
            'lambda_bc':         self.lambda_bc,
            'lambda_res':        self.lambda_res,
            'loss_history':      self.loss_history,
            'loss_ic_history':   self.loss_ic_history,
            'loss_bc_history':   self.loss_bc_history,
            'loss_res_history':  self.loss_res_history,
            'l2_error_history':  self.l2_error_history,
            'lambda_ic_history': self.lambda_ic_history,
            'lambda_bc_history': self.lambda_bc_history,
            'iteration_history': self.iteration_history,
        }, filepath)
        print(f"  Saved → {filepath}")

    def load_model(self, filepath):
        ck = torch.load(filepath, weights_only=False)
        self.load_state_dict(ck['model_state_dict'])
        self.lambda_ic         = ck['lambda_ic']
        self.lambda_bc         = ck['lambda_bc']
        self.lambda_res        = ck['lambda_res']
        self.loss_history      = ck['loss_history']
        self.loss_ic_history   = ck['loss_ic_history']
        self.loss_bc_history   = ck['loss_bc_history']
        self.loss_res_history  = ck['loss_res_history']
        self.l2_error_history  = ck['l2_error_history']
        self.lambda_ic_history = ck.get('lambda_ic_history', [])
        self.lambda_bc_history = ck.get('lambda_bc_history', [])
        self.iteration_history = ck['iteration_history']
        print(f"  Loaded ← {filepath}")

    # ----------------------------------------------------------
    def plot_training_history(self, save_path='results/1d/pinn_training_history.png'):
        """
        4-panel plot reproducing paper Figure 2 plus two diagnostic panels.

        Panel 1 (top-left)  : Total loss vs iteration — log scale (Fig. 2a)
        Panel 2 (top-right) : Adaptive weights lambda_IC, lambda_BC (Fig. 2b)
        Panel 3 (bottom-left) : L2 error with paper/FDM reference lines
        Panel 4 (bottom-right): Individual loss components + phase markers
        """
        iters = np.array(self.iteration_history)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PINN Training History — Fisher-KPP  (Aberqi & Miloudi)",
                     fontsize=13)

        # --- Panel 1: Total loss (Fig. 2a) ---
        axes[0, 0].semilogy(iters, self.loss_history, 'b-', lw=1.5,
                            label='Total loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Loss (log scale)')
        axes[0, 0].set_title('Total Loss vs. Iteration  (Fig. 2a)')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()

        # --- Panel 2: Adaptive weights (Fig. 2b) ---
        if self.lambda_ic_history:
            axes[0, 1].semilogy(iters, self.lambda_ic_history, 'g-',
                                lw=1.5, label='Lambda IC')
            axes[0, 1].semilogy(iters, self.lambda_bc_history, 'b--',
                                lw=1.5, label='Lambda BC')
        axes[0, 1].axhline(self.lambda_max, color='r', ls=':', lw=1.5,
                           label=f'lambda_max = {self.lambda_max:.0f}')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Lambda (log scale)')
        axes[0, 1].set_title('Adaptive Weights vs. Iteration  (Fig. 2b)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # --- Panel 3: L2 error ---
        axes[1, 0].semilogy(iters, self.l2_error_history, 'r-',
                            lw=1.5, label='PINN L2 error')
        axes[1, 0].axhline(1.42e-4, color='k', ls=':', lw=1.5,
                           label='FDM  1.42×10⁻⁴  (paper Table 2)')
        axes[1, 0].axhline(5.57e-2, color='g', ls='--', lw=1.5,
                           label='Paper Phase 1  5.57×10⁻²')
        axes[1, 0].axhline(9.79e-2, color='orange', ls='--', lw=1.5,
                           label='Paper Phase 2/3  9.79×10⁻²')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Relative L2 Error')
        axes[1, 0].set_title('L2 Error vs. Iteration  (Eq. 8)')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(alpha=0.3)

        # --- Panel 4: Loss components ---
        axes[1, 1].semilogy(iters, self.loss_ic_history,
                            label='L_IC (Eq. 2)')
        axes[1, 1].semilogy(iters, self.loss_bc_history,
                            label='L_BC (Eq. 3)')
        axes[1, 1].semilogy(iters, self.loss_res_history,
                            label='L_Res (Eq. 4)')
        for x_mark, lbl, col in [(9999, 'Phase 1 end', 'purple'),
                                   (29999, 'Phase 2 end', 'orange')]:
            if np.any(iters >= x_mark):
                axes[1, 1].axvline(x_mark, color=col, ls='--', lw=1,
                                   label=lbl)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss Components + Phase Markers')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(alpha=0.3)

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved → {save_path}")


def prepare_training_data(data_path):
    """
    Load samples from dataset_generation.py output.

    Column order in the .npz file is [x, t] (x first) — consistent
    with the data generator which does np.hstack([x_col, t_col]).

    FIX-1: returns (x_ic, t_ic, ...) so callers always pass x before t
    to forward(), matching the cat([x, t]) concatenation inside.
    """
    data = np.load(data_path)

    print("\nLoading training data (validating against paper spec)...")
    print(f"  Collocation : {data['collocation'].shape[0]:,} pts  (paper: 10,000)")
    print(f"  Initial     : {data['initial'].shape[0]:,} pts  (paper: 1,000)")
    print(f"  Boundary    : {data['boundary'].shape[0]:,} pts  (paper: 2,000)")

    assert data['collocation'].shape[0] == 10000, "Expected 10,000 collocation pts"
    assert data['initial'].shape[0]     == 1000,  "Expected 1,000 IC pts"
    assert data['boundary'].shape[0]    == 2000,  "Expected 2,000 BC pts"

    def _exact(x, t, D=0.01, R=1.0):
        sqrt_t = np.sqrt(R / (2.0 * D))
        c      = np.sqrt(2.0 * D * R)
        return 1.0 / (1.0 + np.exp(sqrt_t * (x - c * t)))

    # Collocation — col 0 = x, col 1 = t  (matches generator)
    col  = data['collocation'].astype(np.float32)
    x_r  = torch.from_numpy(col[:, 0:1])
    t_r  = torch.from_numpy(col[:, 1:2])

    # Initial condition (t = 0)
    ic   = data['initial'].astype(np.float32)
    x_ic = torch.from_numpy(ic[:, 0:1])
    t_ic = torch.from_numpy(ic[:, 1:2])
    u_ic = torch.from_numpy(_exact(ic[:, 0:1], ic[:, 1:2]).astype(np.float32))

    # Boundary condition (x = 0 and x = 1)
    bc   = data['boundary'].astype(np.float32)
    x_bc = torch.from_numpy(bc[:, 0:1])
    t_bc = torch.from_numpy(bc[:, 1:2])
    u_bc = torch.from_numpy(_exact(bc[:, 0:1], bc[:, 1:2]).astype(np.float32))

    print("  Data validation PASSED — structure matches paper specification")
    return x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_r, t_r


def generate_samples(n_collocation=10000, n_initial=1000,
                     n_boundary=2000, seed=42):
    """Generate training samples (paper Section 3.3)."""
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 70)
    print("DATASET GENERATION (Paper Section 3.3)")
    print("=" * 70)

    x_col = rng.uniform(0.0, 1.0, size=(n_collocation, 1))
    t_col = rng.uniform(0.0, 1.0, size=(n_collocation, 1))
    collocation = np.hstack([x_col, t_col])          # [x, t] columns
    print(f"  Collocation : {collocation.shape[0]:,} pts in [0,1]×[0,1]")

    x_ic = rng.uniform(0.0, 1.0, size=(n_initial, 1))
    t_ic = np.zeros((n_initial, 1))
    initial = np.hstack([x_ic, t_ic])
    print(f"  Initial     : {initial.shape[0]:,} pts at t=0")

    n_b0 = n_boundary // 2
    n_b1 = n_boundary - n_b0
    b0 = np.hstack([np.zeros((n_b0, 1)),
                    rng.uniform(0.0, 1.0, size=(n_b0, 1))])
    b1 = np.hstack([np.ones((n_b1, 1)),
                    rng.uniform(0.0, 1.0, size=(n_b1, 1))])
    boundary = np.vstack([b0, b1])
    print(f"  Boundary    : {boundary.shape[0]:,} pts (x∈{{0,1}}, t∈[0,1])")

    return {'collocation': collocation,
            'initial':     initial,
            'boundary':    boundary}


print("\n" + "=" * 70)
print("DATASET GENERATION")
print("=" * 70)
samples = generate_samples()
out_file = "data_samples.npz"
np.savez_compressed(out_file, **samples)
print(f"\n  Saved → {out_file}")
print(f"  Total points : {sum(v.shape[0] for v in samples.values()):,}")
print("=" * 70)


x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_r, t_r = prepare_training_data(
    "data_samples.npz")

# Move all data to device
x_ic, t_ic, u_ic = x_ic.to(device), t_ic.to(device), u_ic.to(device)
x_bc, t_bc, u_bc = x_bc.to(device), t_bc.to(device), u_bc.to(device)
x_r,  t_r        = x_r.to(device),  t_r.to(device)

pinn = PINN_FisherKPP(
    layers=[2, 50, 50, 50, 50, 50, 50, 50, 1],
    D=0.01, R=1.0,
    adaptive_weights=True,
).to(device)


l2_phase1 = pinn.train_pinn(
    x_ic, t_ic, u_ic,
    x_bc, t_bc, u_bc,
    x_r,  t_r,
    iterations=10_000,
    learning_rate=1e-3,
    decay_rate=0.99,
    scheduler_step_every=100,
    weight_warmup_iters=500,
    print_every=1000,
    phase_name="PHASE 1 — Initial Training  (10k iters, lr=1e-3)",
)
pinn.save_model("pinn_initial.pth")

print(f"\n  Phase 1 L2 = {l2_phase1:.4e}   (paper: 5.57×10⁻²)")


print("\n" + "=" * 70)
print("Phase 2: loading Phase 1 weights  (FIX-8 — paper Section 3.3)")
print("=" * 70)
pinn.load_model("pinn_initial.pth")

l2_phase2 = pinn.train_pinn(
    x_ic, t_ic, u_ic,
    x_bc, t_bc, u_bc,
    x_r,  t_r,
    iterations=20_000,
    learning_rate=1e-4,
    decay_rate=0.99,
    scheduler_step_every=100,
    weight_warmup_iters=500,
    print_every=2000,
    phase_name="PHASE 2 — Retraining 1  (20k iters, lr=1e-4, optimizer reset)",
)
print(f"\n  Phase 2 L2 = {l2_phase2:.4e}   (paper: 9.796×10⁻²)")


l2_phase3 = pinn.train_pinn(
    x_ic, t_ic, u_ic,
    x_bc, t_bc, u_bc,
    x_r,  t_r,
    iterations=20_000,
    learning_rate=1e-4,
    decay_rate=0.99,
    scheduler_step_every=100,
    weight_warmup_iters=500,
    print_every=2000,
    phase_name="PHASE 3 — Retraining 2  (20k iters, lr=1e-4)",
)
pinn.save_model("pinn_final.pth")
print(f"\n  Phase 3 L2 = {l2_phase3:.4e}   (paper: 9.794×10⁻²)")


def run_fdm(D=0.01, R=1.0, Nx=201, T=1.0):
    """
    Explicit forward-Euler FDM (paper Section 3.2, Equation 7).

    dx = 1/(Nx-1) = 0.005
    dt = dx²/(2D)/2 = 0.000625   (half the CFL diffusion limit)
    Nt = T/dt = 1,600 steps

    Expected L2 = 1.42×10⁻⁴  (paper Table 2)
    """
    dx     = 1.0 / (Nx - 1)
    dt_cfl = dx**2 / (2.0 * D)
    dt     = dt_cfl / 2.0
    Nt     = int(round(T / dt))

    def _exact(x, t):
        s = np.sqrt(R / (2.0 * D))
        c = np.sqrt(2.0 * D * R)
        return 1.0 / (1.0 + np.exp(s * (x - c * t)))

    x  = np.linspace(0, 1, Nx)
    u  = _exact(x, 0.0)

    for n in range(Nt):
        tn      = n * dt
        u[0]    = _exact(0.0, tn)
        u[-1]   = _exact(1.0, tn)
        u_xx    = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
        u[1:-1] = u[1:-1] + dt * (D * u_xx + R * u[1:-1] * (1.0 - u[1:-1]))

    u[0]  = _exact(0.0, T)
    u[-1] = _exact(1.0, T)

    u_ex = _exact(x, T)
    l2   = (np.sqrt(np.sum((u - u_ex)**2)) /
            np.sqrt(np.sum(u_ex**2)))
    return l2

print("\n" + "=" * 70)
print("FDM BENCHMARK  (paper Section 3.2)")
print("=" * 70)
l2_fdm = run_fdm()
print(f"  FDM L2 error : {l2_fdm:.4e}")
print(f"  Paper reports: 1.42×10⁻⁴")

# ---- Summary table (paper Table 2) ----
SEP = "=" * 70

# FIX-6: correct FDM reference is 1.42×10⁻⁴, not 9.78×10⁻²
PAPER_FDM    = 1.42e-4
PAPER_PHASE1 = 5.57e-2
PAPER_PHASE2 = 9.7956e-2
PAPER_PHASE3 = 9.7937e-2

print(f"\n{SEP}")
print("FINAL RESULTS SUMMARY  (Paper Table 1 & Table 2)")
print(SEP)
rows = [
    ("FDM  (exact vs FDM)",            l2_fdm,    PAPER_FDM),
    ("PINN Phase 1  (10k iters)",       l2_phase1, PAPER_PHASE1),
    ("PINN Phase 2  (+20k iters)",      l2_phase2, PAPER_PHASE2),
    ("PINN Phase 3  (+20k iters)",      l2_phase3, PAPER_PHASE3),
]
hdr = f"  {'Configuration':<38}  {'Ours':>12}  {'Paper':>12}  {'Ratio':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for label, err, ref in rows:
    flag = "OK" if err / ref < 2.0 else "!!"
    print(f"  [{flag}] {label:<38}  {err:12.4e}  {ref:12.4e}  {err/ref:7.2f}x")

# ---- Key insights from paper Section 5 ----
print(f"\n{SEP}")
print("Paper insights (Section 5):")
print(SEP)
if l2_phase1 < l2_phase2:
    deg = (l2_phase2 - l2_phase1) / l2_phase1 * 100
    print(f"  Retraining degraded performance by {deg:.1f}%")
    print("  Root cause: optimizer state reset disrupts Adam's accumulated")
    print("  momentum — model converges to a higher-error plateau.")
    print("  Paper Section 5.2 recommends saving/restoring optimizer state.")
if l2_fdm < l2_phase1:
    print(f"\n  FDM ({l2_fdm:.2e}) is more accurate than PINN ({l2_phase1:.2e})")
    print("  for this 1D forward problem — consistent with paper findings.")

# ---- Plot ----
pinn.plot_training_history('results/1d/pinn_training_history.png')

# ---- Save ----
np.savez('pinn_results.npz',
         l2_fdm=l2_fdm,
         l2_phase1=l2_phase1,
         l2_phase2=l2_phase2,
         l2_phase3=l2_phase3,
         loss_history=pinn.loss_history,
         l2_error_history=pinn.l2_error_history,
         iteration_history=pinn.iteration_history)

print(f"\n{SEP}")
print("Outputs:")
print("  results/1d/pinn_training_history.png")
print("  pinn_results.npz")
print("  pinn_initial.pth  /  pinn_final.pth")
print(SEP)