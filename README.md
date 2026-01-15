# Healing AI Amnesia in Physics-Informed Neural Networks

This project studies a Physics-Informed Neural Network (PINN) for the 1D Fisher-KPP reaction-diffusion
equation, replicating the setup of Aberqi & Miloudi ([arXiv:2601.11406v1](2601.11406v1.pdf)), with the
goal of investigating whether preserving the Adam optimizer's internal state across retraining phases
changes convergence behavior compared to resetting it each phase (the paper's own protocol).

Work in progress — dataset generation, baseline training, and the optimizer-state comparison will be
added incrementally.
