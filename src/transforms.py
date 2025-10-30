"""
Defines the invertible transformations used in the Normalizing Flow.

Includes:
- Transform (base class)
- PermutationTransform (for shuffling dimensions)
- CouplingTransform (Real NVP-style affine coupling layer)
"""

import torch
from torch import nn
from typing import Optional, Callable

class Transform(nn.Module):
    """Base class for all transform objects."""
    def __init__(self):
        super().__init__()

    def forward(self, inputs, context=None):
        """
        Forward pass (x -> z).
        Must return:
        - transformed inputs
        - log-determinant of the Jacobian
        """
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        """
        Inverse pass (z -> x).
        """
        raise NotImplementedError('InverseNotAvailable')

class PermutationTransform(Transform):
    """
    A transform that permutes the dimensions of the input.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        # Create a random permutation
        permutation = torch.randperm(num_features)
        self.register_buffer('permutation', permutation)
        
        # Create the inverse permutation
        inverse_permutation = torch.argsort(permutation)
        self.register_buffer('inverse_permutation', inverse_permutation)

    def forward(self, inputs: torch.Tensor, context=None):
        """
        Applies the permutation. Log-determinant is 0.
        """
        return inputs[:, self.permutation], 0

    def inverse(self, inputs: torch.Tensor, context=None):
        """
        Applies the inverse permutation.
        """
        return inputs[:, self.inverse_permutation]

class CouplingTransform(Transform):
    """
    An affine coupling layer (Real NVP).
    
    Splits the input features based on a mask. One part is used to
    compute scaling (s) and translation (t) parameters, which are
    then applied to the other part.
    """
    def __init__(
        self,
        num_features: int,
        mask: torch.Tensor,
        hidden_dims: int,
        s_net_factory: Optional[Callable[[int, int, int], nn.Module]] = None,
        t_net_factory: Optional[Callable[[int, int, int], nn.Module]] = None,
    ):
        """
        Args:
            num_features: Dimensionality of the data.
            mask: A binary mask (0s and 1s) of shape (num_features,).
                  Features at 0s are transformed, features at 1s are fixed.
            hidden_dims: Hidden dimension for the s and t networks.
            s_net_factory: Optional custom factory for the scaling network.
            t_net_factory: Optional custom factory for the translation network.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.register_buffer('mask', mask)
        
        # Define default network creation if not provided
        def default_net_factory(input_dim, output_dim, hidden_dims):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, output_dim)
            )
        
        # Use custom factories or defaults
        s_factory = s_net_factory or default_net_factory
        t_factory = t_net_factory or default_net_factory
        
        # We add a Tanh to the scaling network to stabilize training
        self.scaling_net = nn.Sequential(
            s_factory(self.num_features, self.num_features, self.hidden_dims),
            nn.Tanh() # Clamp scale values to [-1, 1] before exp
        )
        self.translation_net = t_factory(
            self.num_features, self.num_features, self.hidden_dims
        )

    def forward(self, x: torch.Tensor, context=None):
        """
        Forward pass: x -> z
        y = x * mask + (x * exp(s) + t) * (1 - mask)
        """
        # Identify fixed and transformed parts
        # masked_x contains only the fixed part
        masked_x = x * self.mask
        
        # Compute s and t from the fixed part
        # Apply (1-mask) to ensure s and t only affect the transformed part
        scale = self.scaling_net(masked_x) * (1 - self.mask)
        translation = self.translation_net(masked_x) * (1 - self.mask)
        
        # Apply the transformation
        # fixed part: y_fixed = x_fixed
        # transformed part: y_transformed = x_transformed * exp(s) + t
        y = masked_x + (x * torch.exp(scale) + translation) * (1 - self.mask)
        
        # Log-determinant of the Jacobian
        # The Jacobian is triangular, so the log-det is the sum of the
        # log-diagonals, which is just the sum of the 'scale' values.
        log_det_jacobian = torch.sum(scale, dim=1)
        
        return y, log_det_jacobian

    def inverse(self, y: torch.Tensor, context=None):
        """
        Inverse pass: z -> x
        x = y * mask + ((y - t) * exp(-s)) * (1 - mask)
        """
        # Identify fixed and transformed parts
        masked_y = y * self.mask
        
        # Compute s and t from the fixed part (which is the same as in forward)
        scale = self.scaling_net(masked_y) * (1 - self.mask)
        translation = self.translation_net(masked_y) * (1 - self.mask)
        
        # Reverse the transformation
        # fixed part: x_fixed = y_fixed
        # transformed part: x_transformed = (y_transformed - t) * exp(-s)
        x = masked_y + (y - translation) * torch.exp(-scale) * (1 - self.mask)
        
        return x
