import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: The total time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process,
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process
        alphas = cosine_schedule(self.num_timesteps)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Store as buffers (will be moved to device when model is moved)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]))
        
        # Precompute useful coefficients
        betas = 1. - alphas
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # For posterior mean
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod))
        # ###########################################################

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Samples from the reverse diffusion process at time t_index.
        Args:
            x: The initial image.
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image.
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        predicted_epsilon = self.model(x, t)
        
        # Extract only the coefficients we actually use
        # alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)  # UNUSED - commented out
        # alphas_cumprod_prev_t = extract(self.alphas_cumprod_prev, t, x.shape)  # UNUSED - commented out
        # betas_t = extract(self.betas, t, x.shape)  # UNUSED - commented out
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t, x.shape)
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        
        # Predict x_0
        x_recon = (x - sqrt_one_minus_alphas_cumprod_t * predicted_epsilon) / sqrt_alphas_cumprod_t
        x_recon = torch.clamp(x_recon, -1., 1.)
        
        # Posterior mean
        posterior_mean = posterior_mean_coef1_t * x_recon + posterior_mean_coef2_t * x
        
        # Posterior variance and sampling
        if t_index == 0:
            # Last step: no noise, just use the posterior mean (z=0 in Algorithm 2)
            return posterior_mean
        else:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Samples from the noise distribution at each time step.
        Args:
            img: The initial image that randomly sampled from the noise distribution.
        Returns:
            The sampled image.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generted image to valid pixel range
        # Hint: to get time index, you can use torch.full()
        img = img  # x_T
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Samples from the noise distribution at each time step.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        ####### TODO: Implement the p_sample function #######
        img = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
        return self.p_sample_loop(img)

    # forward diffusion
    def q_sample(self, x_0, t, noise=None):
        """
        Samples from the noise distribution at time t. Apply alpha interpolation between x_start and noise.
        Args:
            x_0: The initial image.
            t: The time index to sample at.
            noise: The noise tensor to sample from. If None, noise will be sampled.
        Returns:
            The sampled image.
        """

        ####### TODO: Implement the q_sample function #######
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Use pre-computed sqrt buffers (already computed in __init__)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def p_losses(self, x_t, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_t: The noisy image at timestep t.
            t: The time index to compute the loss at.
            noise: The ground truth noise that was added.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        predicted_noise = self.model(x_t, t)
        loss = F.l1_loss(predicted_noise, noise)
        return loss

    def forward(self, x_0, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        loss = self.p_losses(x_t, t, noise)
        return loss
