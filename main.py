"""
Main training script for the 2D Normalizing Flow model.

Trains a Real NVP-style flow model on the 'two-moons' dataset
and generates visualizations of the training process and final results.
"""

import argparse
import logging
import os
import torch
from torch import optim
from tqdm import trange

# Import from our source directory
from src.model import Flow
from src.utils import (
    setup_logging, get_device, load_data, 
    visualize_density, create_gif, plot_comparison
)

# Setup logger for this script
logger = logging.getLogger(__name__)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a 2D Normalizing Flow model.")
    
    # Model Hyperparameters
    parser.add_argument('--num_transforms', type=int, default=5,
                        help="Number of coupling layers in the flow.")
    parser.add_argument('--hidden_dims', type=int, default=128,
                        help="Hidden dimensions for the s and t networks.")
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=150,
                        help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training.")
    
    # Data
    parser.add_argument('--n_samples', type=int, default=1000,
                        help="Number of samples for the 'two-moons' dataset.")
    parser.add_argument('--noise', type=float, default=0.1,
                        help="Noise level for the 'two-moons' dataset.")
    
    # Logging and Output
    parser.add_argument('--log_file', type=str, default='logs/flow.log',
                        help="Path to the log file.")
    parser.add_argument('--frames_dir', type=str, default='frames',
                        help="Directory to save visualization frames.")
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help="Directory to save final GIF and plots.")
    parser.add_argument('--skip_frames', action='store_true',
                        help="If set, skips saving frames during training (faster).")

    return parser.parse_args()

def main():
    """Main function to run the training and evaluation."""
    args = parse_args()
    
    # --- 1. Setup ---
    setup_logging(args.log_file)
    device = get_device()

    # Create output directories
    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 2. Load Data ---
    data_loader, original_data = load_data(
        n_samples=args.n_samples, 
        noise=args.noise, 
        batch_size=args.batch_size
    )
    
    # --- 3. Initialize Model ---
    num_features = 2  # 2D data
    flow = Flow(
        num_features=num_features,
        num_transforms=args.num_transforms,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    optimizer = optim.Adam(flow.parameters(), lr=args.lr)
    
    logger.info("Starting training...")
    logger.info(f"  Epochs:       {args.epochs}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Batch Size:    {args.batch_size}")
    logger.info(f"  Device:       {device}")
    
    # --- 4. Training Loop ---
    flow.train()
    with trange(args.epochs, desc="Training", unit="epoch") as pbar:
        for epoch in pbar:
            total_loss = 0
            for batch in data_loader:
                x, = batch  # DataLoader returns a tuple
                x = x.to(device)
                
                try:
                    # Calculate negative log-likelihood loss
                    loss = -flow.log_prob(x).mean()
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Epoch {epoch}: NaN/Inf loss detected. Skipping batch.")
                        continue

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                except Exception as e:
                    logger.error(f"Error during training step at epoch {epoch}: {e}")
                    continue

            avg_loss = total_loss / len(data_loader)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
            
            # Save visualization frame
            if not args.skip_frames:
                visualize_density(flow, epoch, device, args.frames_dir)

    logger.info("Training complete.")
    
    # --- 5. Final Visualizations ---
    flow.eval()
    
    # a) Final Density Plot
    final_density_path = os.path.join(args.output_dir, 'final_density.png')
    logger.info(f"Saving final density plot to {final_density_path}...")
    visualize_density(flow, args.epochs, device, args.output_dir)
    # Rename the generated frame to a more descriptive name
    os.rename(
        os.path.join(args.output_dir, f'frame_{args.epochs:03d}.png'), 
        final_density_path
    )

    # b) Data Comparison Plot
    logger.info("Generating samples for comparison plot...")
    try:
        samples = flow.sample(num_samples=args.n_samples).cpu()
        comparison_path = os.path.join(args.output_dir, 'data_comparison.png')
        plot_comparison(original_data.cpu(), samples, comparison_path)
    except Exception as e:
        logger.error(f"Failed to generate comparison plot: {e}")

    # c) Create GIF
    if not args.skip_frames:
        gif_path = os.path.join(args.output_dir, 'density_evolution.gif')
        create_gif(args.frames_dir, gif_path)
    else:
        logger.info("Skipping GIF creation as --skip_frames was set.")
        
    logger.info(f"All outputs saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
