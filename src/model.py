"""
Defines the main Normalizing Flow model.

Combines multiple transforms to create a complex, invertible mapping
between a data distribution and a base (Gaussian) distribution.
"""

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Optional

from .transforms import CouplingTransform
from .utils import create_alternating_masks

class Flow(nn.Module):
    """
    A Normalizing Flow model composed of a sequence of invertible transforms.
    """
    def __init__(
        self,
        num_features: int,
        num_transforms: int,
        hidden_dims: int,
        base_distribution: Optional[MultivariateNormal] = None
    ):
        """
        Args:
            num_features: Dimensionality of the data.
            num_transforms: Number of coupling layers to stack.
            hidden_dims: Hidden dimension for the s and t networks.
            base_distribution: The base distribution (e.g., Gaussian).
                               If None, a standard 2D Gaussian is used.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dims = hidden_dims
        
        # Create the alternating masks for the coupling layers
        self.masks = create_alternating_masks(num_features, num_transforms)
        
        # Stack the coupling layers
        self.transforms = nn.ModuleList([
            CouplingTransform(self.num_features, mask, self.hidden_dims) 
            for mask in self.masks
        ])
        
        # Set the base distribution
        if base_distribution is None:
            self.base_distribution = MultivariateNormal(
                loc=torch.zeros(self.num_features),
                covariance_matrix=torch.eye(self.num_features)
            )
        else:
            self.base_distribution = base_distribution

    def forward(self, x: torch.Tensor):
        """
        Forward pass (data to latent): x -> z
        Applies all transforms in sequence and sums their log-det-Jacobians.
        """
        log_det_jacobian_total = 0
        z = x
        for transform in self.transforms:
            z, log_det_jacobian = transform(z)
            log_det_jacobian_total += log_det_jacobian
            
        return z, log_det_jacobian_total

    def inverse(self, z: torch.Tensor):
        """
        Inverse pass (latent to data): z -> x
        Applies all transforms in reverse order.
        """
        x = z
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
            
        return x

    def log_prob(self, x: torch.Tensor):
        """
        Computes the log-probability of the data x using the
        change of variables formula.
        
        log p_X(x) = log p_Z(f(x)) + log |det J_f(x)|
        """
        z, log_det_J = self.forward(x)
        # Move base distribution to the same device as z
        base_dist = self.base_distribution
        if base_dist.loc.device != z.device:
            base_dist = MultivariateNormal(
                loc=base_dist.loc.to(z.device),
                covariance_matrix=base_dist.covariance_matrix.to(z.device)
            )
        
        log_prob_z = base_dist.log_prob(z)
        log_prob_x = log_prob_z + log_det_J
        
        return log_prob_x

    def sample(self, num_samples: int):
        """
        Generates new samples from the model by sampling from the
        base distribution and applying the inverse transformation.
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Move base distribution to the correct device
            base_dist = self.base_distribution
            if base_dist.loc.device != device:
                base_dist = MultivariateNormal(
                    loc=base_dist.loc.to(device),
                    covariance_matrix=base_dist.covariance_matrix.to(device)
                )
                
            base_samples = base_dist.sample((num_samples,))
            samples = self.inverse(base_samples)
            
        return samples

    def sample_and_log_prob(self, num_samples: int):
        """
        Samples from the model and computes the log-probability
        of the generated samples.
        """
        x = self.sample(num_samples)
        log_prob = self.log_prob(x)
        
        return x, log_prob
