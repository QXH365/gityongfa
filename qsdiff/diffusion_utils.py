# diffusion_utils.py

import numpy as np
import torch
import torch.nn as nn

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    Generates a variance schedule for the diffusion process.
    This function is adapted from your provided code.
    """
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    else:
        raise NotImplementedError(beta_schedule)
    return betas

class NoiseScheduler(nn.Module):
    """
    Manages the noise schedule and forward process (adding noise to data).
    It pre-calculates all the necessary constants (alphas, betas, etc.) for efficiency.
    """
    def __init__(self, schedule_type='linear', timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        
        self.timesteps = timesteps
        
        # 1. Generate beta schedule
        betas = get_beta_schedule(
            beta_schedule=schedule_type,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=timesteps,
        )
        betas = torch.from_numpy(betas).float()

        # 2. Pre-calculate alphas and other constants for the diffusion formula
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # These are the terms used in the q(x_t | x_0) formula
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Register these constants as buffers, so they are part of the module's state
        # but are not considered learnable parameters.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def _get_t(self, tensor, t):
        """Helper function to extract the value at timestep t and reshape it."""
        return tensor.gather(-1, t).reshape(-1, 1)

    def add_noise(self, original_pos, noise, timesteps):
        """
        Performs the forward diffusion process.
        Calculates q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            original_pos (torch.Tensor): The ground truth coordinates (x_0). Shape: [N, 3]
            noise (torch.Tensor): A standard Gaussian noise tensor. Shape: [N, 3]
            timesteps (torch.Tensor): A tensor of timesteps for each atom. Shape: [N]

        Returns:
            torch.Tensor: The noisy coordinates at the given timesteps (x_t). Shape: [N, 3]
        """
        # Get the constants for the given timesteps
        sqrt_alpha_cumprod_t = self._get_t(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alpha_cumprod_t = self._get_t(self.sqrt_one_minus_alphas_cumprod, timesteps)

        # Apply the forward process formula
        noisy_pos = (
            sqrt_alpha_cumprod_t * original_pos + 
            sqrt_one_minus_alpha_cumprod_t * noise
        )
        return noisy_pos