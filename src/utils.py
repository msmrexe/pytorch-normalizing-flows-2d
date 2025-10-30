"""
Utility functions for the Normalizing Flows project.

Includes:
- Logging setup
- Device selection
- Data loading (Two Moons)
- Mask creation
- Visualization (density, comparison plot, GIF)
"""

import os
import logging
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from typing import List
from IPython.display import Image, display

# Setup a logger
logger = logging.getLogger(__name__)

def setup_logging(log_file: str = 'logs/flow.log'):
    """Configures logging to file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logger.info("Logging configured.")

def get_device() -> torch.device:
    """Selects the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device

def load_data(n_samples: int = 1000, noise: float = 0.1, batch_size: int = 64) -> DataLoader:
    """
    Generates the 'two moons' dataset and returns a DataLoader.
    
    Args:
        n_samples: Number of data points to generate.
        noise: Noise level for the dataset.
        batch_size: Batch size for the DataLoader.

    Returns:
        A DataLoader for the 'two moons' dataset.
    """
    logger.info(f"Generating {n_samples} 'two moons' samples with noise={noise}...")
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info("Dataset loaded successfully.")
    return data_loader, X_tensor

def create_alternating_masks(num_features: int, num_masks: int) -> List[torch.Tensor]:
    """
    Generates a list of alternating binary masks.
    
    Args:
        num_features: The dimensionality of the data (e.g., 2 for 2D).
        num_masks: The number of masks to create (e.g., number of transforms).

    Returns:
        A list of torch.Tensor masks.
    """
    masks = []
    for i in range(num_masks):
        start = i % 2
        # Create a mask [start, 1-start, start, 1-start, ...]
        mask_pattern = [(start + j) % 2 for j in range(num_features)]
        masks.append(torch.tensor(mask_pattern, dtype=torch.float32))
    logger.info(f"Created {num_masks} alternating masks.")
    return masks

def visualize_density(flow: torch.nn.Module, epoch: int, device: torch.device, output_dir: str = 'frames'):
    """
    Visualizes the learned density of the flow model at a given epoch
    and saves it as a PNG frame.
    
    Args:
        flow: The trained flow model.
        epoch: The current epoch number (for file naming).
        device: The device to run computations on.
        output_dir: Directory to save the frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    flow.eval() # Set model to evaluation mode
    
    with torch.no_grad():
        # Create a grid over the range of the data
        xline = torch.linspace(-2, 3, 300, device=device)
        yline = torch.linspace(-1, 1.5, 300, device=device)
        xgrid, ygrid = torch.meshgrid(xline, yline, indexing='ij')
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        # Compute the log probability for each point in the grid
        log_prob_grid = flow.log_prob(xyinput)
        zgrid = torch.exp(log_prob_grid).reshape(300, 300).cpu()

        # Plot the density
        plt.figure(figsize=(8, 6))
        plt.contourf(
            xgrid.cpu().numpy(),
            ygrid.cpu().numpy(),
            zgrid.numpy(),
            levels=50,
            cmap='viridis'
        )
        plt.title(f'Learned Density at Epoch {epoch}')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.colorbar(label='Density')

        # Save frame as an image
        filename = os.path.join(output_dir, f'frame_{epoch:03d}.png')
        plt.savefig(filename)
        plt.close()
    flow.train() # Set model back to training mode

def create_gif(frames_dir: str = 'frames', gif_path: str = 'outputs/density_evolution.gif', fps: int = 10):
    """
    Creates a GIF from the saved frames.
    
    Args:
        frames_dir: Directory containing the saved .png frames.
        gif_path: Path to save the final GIF.
        fps: Frames per second for the GIF.
    """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    logger.info(f"Creating GIF... saving to {gif_path}")
    images = []
    try:
        filenames = sorted(os.listdir(frames_dir))
        for frame in filenames:
            if frame.endswith(".png"):
                images.append(imageio.imread(os.path.join(frames_dir, frame)))
        
        if not images:
            logger.warning(f"No .png frames found in {frames_dir}. GIF not created.")
            return

        imageio.mimsave(gif_path, images, fps=fps)
        logger.info("GIF created successfully.")
        # Display the GIF in a notebook environment
        try:
            display(Image(filename=gif_path))
        except Exception:
            pass # Fails if not in a notebook
            
    except Exception as e:
        logger.error(f"Failed to create GIF: {e}")

def plot_comparison(original_data: torch.Tensor, generated_samples: torch.Tensor, save_path: str = 'outputs/data_comparison.png'):
    """
    Plots a scatter plot comparing original data and generated samples.
    
    Args:
        original_data: The original 'two moons' data.
        generated_samples: Samples from the trained flow model.
        save_path: Path to save the comparison plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f"Generating comparison plot... saving to {save_path}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(original_data[:, 0], original_data[:, 1], label="Original Data", alpha=0.5, s=10)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label="Generated Data", alpha=0.5, s=10)
    plt.legend()
    plt.title("Original vs. Generated Data")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(save_path)
    plt.close()
    logger.info("Comparison plot saved.")
