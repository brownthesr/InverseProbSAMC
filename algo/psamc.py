import torch
from illuma_samc import SAMCWeights, UniformPartition
from illuma_samc.gain import GainSequence
from utils.scheduler import Scheduler
from tqdm import tqdm
from .base import Algo

import wandb

# ----------------------------------------------------------------------------------------
# Paper: Ensemble kalman diffusion guidance: A derivative-free method for inverse problems
# Official implementation: https://github.com/devzhk/enkg-pytorch
# ----------------------------------------------------------------------------------------

class PSAMC(Algo):
    def __init__(self, 
                 net, 
                 forward_op,
                 guidance_scale, 
                 samc_iterations, 
                 diffusion_scheduler_config,
                 num_samples=1024,
                 batch_size=128,
                 alpha=1,
                 gamma1=1,
                 t0=1,
                 n_bins=40,
                 theta=torch.pi/6,
                 p1=.5,
                 p2=.5,

        ):
        super(PSAMC, self).__init__(net, forward_op)
        # total = 240_000
        # samc_iteration_range = (15,1000)
        # num_samples_range = (2, 256)
        # num_steps_range = (50, 1000)
        #
        # samc_iterations = int(samc_iteration_range[0]+p1*(samc_iteration_range[1]-samc_iteration_range[0]))
        # num_samples_allow_min = max(num_samples_range[0],total/(samc_iterations*num_steps_range[1]))
        # num_samples_allow_max = min(num_samples_range[1],total/(samc_iterations*num_steps_range[0]))
        #
        # num_samples = int(num_samples_allow_min + p2*(num_samples_allow_max - num_samples_allow_min))
        # diffusion_scheduler_config.num_steps = int(total/(num_samples * samc_iterations))
        #
        # print(f"num_samples {num_samples}, samc_iterations: {samc_iterations}, diffusion_steps: {diffusion_scheduler_config.num_steps}")
        self.samc_iterations = samc_iterations
        print(samc_iterations)
        if batch_size > num_samples:
            batch_size = num_samples
        self.batch_size = batch_size

        self.num_samples = num_samples
        self.scheduler = Scheduler(**diffusion_scheduler_config)

        self.theta = torch.tensor([theta])
        self.beta = guidance_scale
        # gain_kwargs = {"rho": 1, "tau":1, "warmup": 1, "step_scale":samc_iterations//2}
        gain_kwargs = {"t0": t0, "gamma1":gamma1*samc_iterations, "alpha":alpha}
        # gain = GainSequence("ramp", **gain_kwargs)
        gain = GainSequence("1/t", **gain_kwargs)
        self.wm = SAMCWeights(partition=UniformPartition(0,2.5,n_bins), record_every=2, gain=gain)

    @torch.no_grad()
    def get_energy(self, z, observation):
        x0s = torch.zeros_like(z)  # (N, C, H, W), x0 of each particle
        num_batches = z.shape[0] // self.batch_size
        # get x0 for each particle
        for i in range(num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            x0s[start:end] = self.ode_sampler_dps(
                z[start:end],
            )

        energy = self.forward_op.loss(x0s, observation)
        return energy.flatten()

    @torch.no_grad()
    def propose_updates(self, it, z, energies, observation):

        B, C, H, W = z.shape

        # Create pCN proposal
        xi = torch.randn_like(z,device=self.forward_op.device)*self.scheduler.sigma_max
        theta = torch.tensor(self.theta, device=self.forward_op.device)
        z_prop = torch.cos(theta)*z+torch.sin(theta)*xi
        E_prop = self.get_energy(z_prop,observation)

        # Compute Acceptance probability
        log_alpha = self.beta * (energies -E_prop) +self.wm.correction(energies.to("cpu"), E_prop.to("cpu")).to("cuda")     # (K, B)
        accept = torch.log(torch.rand_like(log_alpha)) < log_alpha  # (K, B)
        accept_mask = accept.view(B, 1, 1, 1)
        z_new = torch.where(accept_mask, z_prop, z)
        energies_new = torch.where(accept, E_prop, energies)

        # Step our bins
        self.wm.step(it, energies_new.to("cpu"))

        # Compute some metrics and return the final value
        accept_ratio = accept.float().mean().item()  # scalar
        return z_new, energies_new, accept_ratio


    @torch.no_grad()
    def inference(self, observation, num_samples=1):
        device = self.forward_op.device
        x_initial = torch.randn(self.num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max

        num_batches = x_initial.shape[0] // self.batch_size
        # Things are (n,*, *, *)
        # Starting with 80 samc_iterations and a batch size of 2

        # Main MCMC loop
        alpha_accept = .1
        z = x_initial
        pbar = tqdm(total=self.samc_iterations, desc="samc_iterations")
        acc_rate_ema = None
        energies = self.get_energy(z, observation)
        for it in range(self.samc_iterations):
            z, energies, acceptance_ratio = self.propose_updates(it, z, energies, observation)

            if acc_rate_ema is None:
                acc_rate_ema = acceptance_ratio
            else :
                acc_rate_ema = alpha_accept * acceptance_ratio + (1 - alpha_accept) * acc_rate_ema
            pbar.set_postfix(ema_acc=f"{acc_rate_ema:.3f}", cur_acc=f"{acceptance_ratio:.3f}")
            pbar.update(1)

        x0s = torch.zeros_like(z)  # (N, C, H, W), x0 of each particle
        # get x0 for each particle
        for i in range(num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            x0s[start:end] = self.ode_sampler_dps(
                z[start:end],
            )
        self.wm.plot_diagnostics(save_path=f"checker.png")
        # Set up returning final batch of samples.
        idx = torch.argmin(energies)
        return x0s[idx].unsqueeze(0)


# ----------- deterministic sampler ------------#
# Generate x_0 from x_t for any t.
    @torch.no_grad()
    def ode_sampler_dps(
        self, # Assuming this lives in the same class as the scheduler
        x_initial,
        num_samples=1,
    ):
        device = self.forward_op.device
        x_next = x_initial
        
        # We use the range from the scheduler just like the inference script
        for i in range(self.scheduler.num_steps):
            x_cur = x_next
            
            # 1. Extract scheduler constants for this step
            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]
            s_step = self.scheduler.scaling_steps[i]

            # 2. Denoising pass (Unconditional)
            # Identical to: denoised = self.net(x_cur / s_step, torch.as_tensor(sigma)...)
            denoised = self.net(
                x_cur / s_step, 
                torch.as_tensor(sigma).to(x_cur.device)
            )

            # 3. Score calculation
            # Identical to the logic in your second image
            score = (denoised - x_cur / s_step) / (sigma ** 2) / s_step

            # 4. Deterministic Update
            # This is the "else" branch (non-SDE) from your script
            # x_next = x_cur * scaling_factor + factor * score * 0.5
            x_next = x_cur * scaling_factor + factor * score * 0.5

        return x_next

@torch.no_grad()
def ode_sampler(
    net,
    x_initial,
    num_steps=18,
    sigma_start=80.0,
    sigma_eps=0.002,
    rho=7,
):
    if num_steps == 1:
        denoised = net(x_initial, sigma_start)
        return denoised
    last_sigma = sigma_eps
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_initial.device)

    t_steps = (
        sigma_start ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (last_sigma ** (1 / rho) - sigma_start ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = x_initial
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        t_hat = t_cur
        x_hat = x_cur

        # Euler step.
        denoised = net(x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next

