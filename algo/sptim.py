import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler

from time import time

class SPTIM(Algo):
    """
    SPT_iterative_max: Ensemble diffusion sampling with per‑chain source‑steering.

    For each chain independently:
      - Perform unconditional Langevin steps on all particles.
      - Pick the particle with the *maximum* forward loss (worst data consistency).
      - Optionally maximize its loss further via gradient ascent.
      - Compute the DPS gradient from this particle and apply it to every
        particle in the same chain.
    """

    def __init__(self,
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 num_samples,
                 sde=True,
                 num_langevin_steps=1,   # Unconditional steps per noise level
                 inner_opt_steps=0,      # Gradient ascent steps for best particle
                 inner_opt_lr=0.01,      # LR for inner loss maximization
                 second_order=True,      # Whether to keep graph in inner loop
                 n_chains=1,             # Number of parallel chains (ensemble size)
                 max_beta=5.0,             # Temperature schedule for swapping
                 step_size=0.1,          # Langevin step size
                 device=None):           # Device to run on
        super().__init__(net, forward_op)
        self.scale = guidance_scale
        self.scheduler = Scheduler(**diffusion_scheduler_config)  # assumed defined elsewhere
        self.sde = sde
        self.num_samples = num_samples
        self.num_langevin_steps = num_langevin_steps
        self.inner_opt_steps = inner_opt_steps
        self.inner_opt_lr = inner_opt_lr
        self.second_order = second_order

        # Chain handling attributes
        self.n_chains = n_chains
        self.betas = torch.linspace(0.0, max_beta, self.n_chains, device=device)
        self.step_size = step_size
        self.device = device if device is not None else next(net.parameters()).device

    @torch.no_grad()
    def get_energy(self, z, sigma, scaling, observation):
        """
        Compute energy using the Tweedie estimate:
            E(z) = loss( denoiser(z/scaling, sigma), observation )
        z: [n_chains, batch_size, ...]   (typically [n_chains, num_samples, C, H, W])
        sigma: float (current noise level)
        scaling: float (normalization factor)
        observation: [batch_size, ...]   (already repeated to match num_samples)
        Returns: [n_chains, batch_size]
        """
        # Flatten chain and sample dimensions for the network
        flat_z = z.view(-1, *z.shape[2:])  # [n_chains * num_samples, C, H, W]

        # Tweedie estimate
        x_hat = self.net(flat_z / scaling, torch.as_tensor(sigma, device=z.device))
        x_hat = x_hat.view(z.shape[0], z.shape[1], *x_hat.shape[1:])  # back to [n_chains, num_samples, ...]

        # Expand observation to match x_hat's shape
        # observation has shape [num_samples, C, H, W] (repeated if needed)
        # We need to repeat it n_chains times along a new dimensio
        # Flatten x_hat to compute loss per sample
        x_flat = x_hat.reshape(-1, *x_hat.shape[2:])  # [n_chains*num_samples, C, H, W]
        # Compute loss (per sample)
        losses = self.forward_op.loss(x_flat, observation[0])  # shape [n_chains*num_samples]
        # Reshape back to [n_chains, num_samples]
        return losses.view(z.shape[0], z.shape[1])
    @torch.no_grad()
    def swap_between_chains(self, z, energies):
        """Swap particles between chains based on Metropolis‑Hastings."""
        for k in range(self.n_chains - 1):
            delta_beta = self.betas[k] - self.betas[k + 1]
            delta_E = energies[k + 1] - energies[k]
            log_alpha = delta_beta * delta_E
            accept = torch.rand(z.shape[1], device=self.device).log() < log_alpha
            if accept.any():
                z_k, E_k = z[k].clone(), energies[k].clone()
                z[k, accept] = z[k + 1, accept]
                energies[k, accept] = energies[k + 1, accept]
                z[k + 1, accept] = z_k[accept]
                energies[k + 1, accept] = E_k[accept]
        return z, energies

    @torch.inference_mode()
    def propose_updates(self, z, sigma, scaling, observation):

        flat_z = z.reshape(-1, *z.shape[2:])
        sigma_t = torch.as_tensor(sigma, device=z.device, dtype=z.dtype)

        denoised = self.net(flat_z / scaling, sigma_t)
        denoised = denoised.reshape_as(z)

        # Score for Langevin update
        score = (denoised - z / scaling) / (sigma * sigma * scaling)

        beta = self.betas.view(-1, 1, 1, 1, 1)
        z.add_(0.5 * self.step_size * beta * score)
        z.add_(torch.randn_like(z).mul_(self.step_size ** 0.5))

        # Energy estimate from the SAME forward pass
        losses = self.forward_op.loss(
            denoised.reshape(-1, *denoised.shape[2:]),
            observation[0],
        )
        energies = losses.view(z.shape[0], z.shape[1])

        return z, energies

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        self.betas = self.betas.to(device)

        # Initialize particles as noisy images at sigma_max
        x = torch.randn(self.n_chains, self.num_samples, self.net.img_channels,
                        self.net.img_resolution, self.net.img_resolution,
                        device=device) * self.scheduler.sigma_max
        x_next = x.clone()

        # Initial energies using the first sigma
        energies = self.get_energy(x, self.scheduler.sigma_steps[0],
                                self.scheduler.scaling_factor[0], observation)

        pbar = tqdm(range(self.scheduler.num_steps))
        for i in pbar:
            sigma = self.scheduler.sigma_steps[i]
            scaling = self.scheduler.scaling_factor[i]
            inv_scaling = self.scheduler.scaling_steps[i]

            # 1. Unconditional diffusion (Langevin) updates on all particles
            #    No gradients needed for these sampling steps.
            start = time()
            with torch.inference_mode():
                x_cur = x_next
                for _ in range(self.num_langevin_steps):
                    x_cur, energies = self.propose_updates(x_cur, sigma, scaling, observation)

                # swap once per noise level instead of every ULA step
                x_cur, energies = self.swap_between_chains(x_cur, energies)
            print(f"ULA: {time()-start}")
            # 2. Create a fresh leaf tensor with gradient for the DPS correction
            x_cur_grad = x_cur.detach().requires_grad_(True)   # [n_chains, num_samples, C, H, W]

            # 3. Per‑chain DPS correction (memory‑efficient)
            x_next = torch.empty_like(x_cur_grad)   # allocate new tensor for updated state
            min_loss_overall = float('inf')

            for c in range(self.n_chains):
                # --- Extract current chain's data (requires grad) ---
                x_cur_chain = x_cur_grad[c]                 # [num_samples, C, H, W]

                # --- Forward pass for all samples in this chain (no gradients for selection) ---
                with torch.no_grad():
                    flat_x_cur = x_cur_chain.view(-1, *x_cur_chain.shape[1:])  # [num_samples, C, H, W]
                    denoised_all = self.net(flat_x_cur / inv_scaling,
                                            torch.as_tensor(sigma, device=device))
                    denoised_all = denoised_all.view(-1, *denoised_all.shape[1:])  # [num_samples, C, H, W]
                    losses = self.forward_op.loss(denoised_all, observation[0])    # [num_samples]

                # --- Find best sample (lowest loss) in this chain ---
                best_idx = torch.argmin(losses)
                best_x = x_cur_chain[best_idx:best_idx+1]               # [1, C, H, W]
                best_denoised = denoised_all[best_idx:best_idx+1]       # [1, C, H, W]

                # --- Optional inner optimization on the best denoised image ---
                if self.inner_opt_steps > 0:
                    best_denoised_opt = best_denoised.detach().requires_grad_(True)
                    for _ in range(self.inner_opt_steps):
                        loss_val = self.forward_op.loss(best_denoised_opt, observation).sum()
                        grad_denoised = torch.autograd.grad(loss_val, best_denoised_opt,
                                                            create_graph=self.second_order)[0]
                        best_denoised_opt = (best_denoised_opt + self.inner_opt_lr * grad_denoised).detach()
                        if self.second_order:
                            best_denoised_opt.requires_grad_(True)
                    denoised_for_corr = best_denoised_opt
                else:
                    denoised_for_corr = best_denoised

                # --- Compute DPS gradient for the best noisy image ---
                best_x_scaled = best_x / inv_scaling
                denoised_recomputed = self.net(best_x_scaled,
                                            torch.as_tensor(sigma, device=device))

                grad_forward, loss_scale = self.forward_op.gradient(denoised_for_corr, observation,
                                                                    return_loss=True)

                # Use create_graph=False unless second‑order gradients are required
                ll_grad = torch.autograd.grad(denoised_recomputed, best_x, grad_forward,
                                            create_graph=False)[0]
                ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)

                # --- Apply correction to all samples in this chain ---
                ll_grad_full = ll_grad.expand(self.num_samples, -1, -1, -1)  # [num_samples, C, H, W]
                x_next_chain = x_cur_chain - self.scale * ll_grad_full
                x_next[c] = x_next_chain

                # --- Update minimum loss for monitoring ---
                min_loss_overall = min(min_loss_overall, losses.min().item())

                # --- Clean up to free memory early ---
                del x_cur_chain, flat_x_cur, denoised_all, losses
                del best_x, best_denoised, best_x_scaled, denoised_recomputed
                del grad_forward, ll_grad, ll_grad_full, x_next_chain
                if self.inner_opt_steps > 0:
                    del best_denoised_opt, grad_denoised

            # Update progress bar
            pbar.set_description(f'Step {i+1}/{self.scheduler.num_steps} | Min loss:{min_loss_overall:.4f}')

        # Return the last chain (or you could return the chain with lowest loss)
        return x_next[-1]