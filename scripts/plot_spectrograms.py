#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.nn import functional as F

from emg2qwerty.data import EMGSessionData
from emg2qwerty.lightning import AutoencoderModule
from emg2qwerty.transforms import LogSpectrogram


def load_checkpoint(checkpoint_path: str) -> AutoencoderModule:
    """Load a trained autoencoder from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model = AutoencoderModule()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def compute_spectrogram(emg_data: np.ndarray, n_fft: int = 64, hop_length: int = 32) -> np.ndarray:
    """Compute spectrogram from raw EMG data."""
    # Convert numpy array to torch tensor
    emg_tensor = torch.from_numpy(emg_data).float()

    # Process each channel separately
    specs = []
    for ch in range(emg_data.shape[1]):
        # Extract single channel
        channel_data = emg_tensor[:, ch]

        # Use the LogSpectrogram transform from transforms.py
        transform = LogSpectrogram(n_fft=n_fft, hop_length=hop_length)

        # Apply transform to single channel (reshape to [time, 1])
        channel_data = channel_data.reshape(-1, 1)
        spec = transform(channel_data)  # Shape: [time, 1, freq]

        # Remove the singleton dimension and get the result
        spec = spec.squeeze(1)  # Shape: [time, freq]

        # Transpose to get [freq, time]
        spec = spec.T

        specs.append(spec.numpy())

    # Stack all channel spectrograms
    return np.stack(specs)


def load_sample_data(
    data_path: str, num_samples: int = 3, window_length: int = 2000
) -> List[torch.Tensor]:
    """Load a few sample windows from an HDF5 file and compute spectrograms."""
    with EMGSessionData(Path(data_path)) as session:
        # Get total length of the session
        session_length = len(session)

        # Choose random starting points for windows
        if session_length <= window_length:
            # If session is shorter than window length, just use the whole session
            indices = [0]
            window_length = session_length
        else:
            # Choose random starting points
            max_start_idx = session_length - window_length
            indices = np.random.choice(max_start_idx, num_samples, replace=False)

        samples = []
        for idx in indices:
            # Get a window of raw EMG data
            window = session[idx : idx + window_length]

            # Extract left and right EMG data
            emg_left = window[EMGSessionData.EMG_LEFT]
            emg_right = window[EMGSessionData.EMG_RIGHT]

            # Combine left and right EMG data
            emg_combined = np.concatenate([emg_left, emg_right], axis=1)

            # Compute spectrogram
            spec = compute_spectrogram(emg_combined)

            # Convert to torch tensor
            spec_tensor = torch.from_numpy(spec).float()

            # # Apply normalization similar to what would happen in the dataloader
            # transform = Compose([LogSpectrogram()])
            # spec_tensor = transform({"inputs": spec_tensor})["inputs"]

            samples.append(spec_tensor)

        return samples


def plot_spectrograms(
    original: torch.Tensor, reconstructed: torch.Tensor, output_dir: str, sample_idx: int = 0
) -> None:
    """Plot original and reconstructed spectrograms side by side."""
    # Ensure tensors are in the right format
    # If reconstructed has 4 dimensions (batch, hands, channels, freq), reshape it
    if reconstructed.dim() == 4:
        # Reshape to match original format
        reconstructed = reconstructed.squeeze(0)  # Remove batch dimension
    elif reconstructed.dim() == 5:  # If shape is [batch, hands, channels, freq, time]
        batch, hands, channels, freq, time = reconstructed.shape
        # Reshape to match original format - combining hands and channels
        reconstructed = reconstructed.squeeze(0)  # Remove batch dimension
        reconstructed = reconstructed.reshape(hands * channels, freq, time)

    # Handle original tensor similarly
    if original.dim() == 3:  # If original is (channels, freq, time)
        pass  # Already in the right format
    elif original.dim() == 4:  # If original is (batch, channels, freq, time)
        original = original.squeeze(0)  # Remove batch dimension

    # Get the number of channels
    n_channels = original.shape[0]

    # Create a figure with two columns (original and reconstructed)
    # If there are many channels, arrange them in a grid
    n_rows = min(8, n_channels)  # Maximum 8 rows
    n_cols = 2 * ((n_channels + n_rows - 1) // n_rows)  # Ceiling division for columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))

    # Set the figure title
    fig.suptitle(f"Original vs Reconstructed Spectrograms (Sample {sample_idx})")

    # Flatten axes if it's a 1D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each channel
    for i in range(n_channels):
        row = i % n_rows
        col = 2 * (i // n_rows)

        # Original spectrogram
        im0 = axes[row, col].imshow(original[i].numpy(), aspect="auto", cmap="viridis")
        axes[row, col].set_title(f"Original Ch {i}")

        # Reconstructed spectrogram - handle potential shape mismatch
        if i < reconstructed.shape[0]:
            im1 = axes[row, col + 1].imshow(reconstructed[i].numpy(), aspect="auto", cmap="viridis")
            axes[row, col + 1].set_title(f"Reconstructed Ch {i}")
        else:
            axes[row, col + 1].text(0.5, 0.5, "No reconstruction", ha="center", va="center")
            axes[row, col + 1].set_title(f"Missing Ch {i}")
            im1 = im0  # Use original for colorbar

        # Add colorbar for the first row
        if row == 0:
            fig.colorbar(im0, ax=axes[row, col], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[row, col + 1], fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(n_channels, n_rows * (n_cols // 2)):
        row = i % n_rows
        col = 2 * (i // n_rows)
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for the title

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"spectrogram_comparison_{sample_idx}.png"), dpi=300)
    plt.close()


def plot_bottleneck(bottleneck: torch.Tensor, output_dir: str, sample_idx: int = 0) -> None:
    """Plot the bottleneck representation."""
    # Get the number of channels in the bottleneck
    n_channels = bottleneck.shape[0]

    # Create a figure
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2 * n_channels))

    # Set the figure title
    fig.suptitle(f"Bottleneck Representation (Sample {sample_idx})")

    # If there's only one channel, axes won't be an array
    if n_channels == 1:
        axes = [axes]

    # Plot each channel
    for i in range(n_channels):
        im = axes[i].imshow(bottleneck[i].numpy(), aspect="auto", cmap="viridis")
        axes[i].set_ylabel(f"Channel {i}")

        # Add colorbar
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for the title

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"bottleneck_representation_{sample_idx}.png"), dpi=300)
    plt.close()


def plot_reconstruction_error(
    original: torch.Tensor, reconstructed: torch.Tensor, output_dir: str, sample_idx: int = 0
) -> None:
    """Plot the reconstruction error."""
    # Calculate the error
    error = F.mse_loss(reconstructed, original, reduction="none")

    # Get the number of channels
    n_channels = original.shape[0]

    # Create a figure
    # If there are many channels, arrange them in a grid
    n_rows = min(8, n_channels)  # Maximum 8 rows
    n_cols = (n_channels + n_rows - 1) // n_rows  # Ceiling division for columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))

    # Set the figure title
    fig.suptitle(f"Reconstruction Error (Sample {sample_idx})")

    # Make axes a 2D array if it's 1D or a single subplot
    if n_channels == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each channel
    for i in range(n_channels):
        row = i % n_rows
        col = i // n_rows

        im = axes[row, col].imshow(error[i].numpy(), aspect="auto", cmap="hot")
        axes[row, col].set_title(f"Channel {i}")

        # Add colorbar
        fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(n_channels, n_rows * n_cols):
        row = i % n_rows
        col = i // n_rows
        axes[row, col].axis("off")

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for the title

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"reconstruction_error_{sample_idx}.png"), dpi=300)
    plt.close()


def main(
    checkpoint: str = typer.Argument(..., help="Path to the autoencoder checkpoint"),
    data: str = typer.Argument(..., help="Path to an HDF5 file with EMG data"),
    output_dir: str = typer.Option("spectrogram_plots", help="Directory to save plots"),
    num_samples: int = typer.Option(3, help="Number of samples to plot"),
    window_length: int = typer.Option(2000, help="Length of EMG window to use (in samples)"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
) -> None:
    """
    Plot spectrograms before and after autoencoder reconstruction.

    This script loads a trained autoencoder model from a checkpoint,
    processes a few sample EMG windows into spectrograms, and visualizes
    the original spectrograms, their reconstructions, the bottleneck
    representations, and the reconstruction errors.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load the model
    typer.echo(f"Loading model from checkpoint: {checkpoint}")
    model = load_checkpoint(checkpoint)

    # Load sample data
    typer.echo(f"Loading {num_samples} samples from: {data}")
    samples = load_sample_data(data, num_samples, window_length)

    # Process each sample
    for i, sample in enumerate(samples):
        typer.echo(f"Processing sample {i + 1}/{len(samples)}")

        # Add batch dimension
        sample_batch = sample.unsqueeze(0)

        # Forward pass through the autoencoder
        with torch.no_grad():
            bottleneck, reconstructed = model(sample_batch)

        # Print shapes for debugging
        print(f"Sample shape: {sample.shape}")
        print(f"Bottleneck shape: {bottleneck.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")

        # Reshape reconstructed to match original format
        # The model seems to be treating time as batch dimension
        if reconstructed.dim() == 5:  # [time, batch, hands, channels, freq]
            time, batch, hands, channels, freq = reconstructed.shape
            # Reshape to [channels*hands, freq, time]
            reconstructed_reshaped = reconstructed.permute(2, 3, 4, 0, 1).reshape(
                hands * channels, freq, time
            )

            # Plot the spectrograms
            plot_spectrograms(sample, reconstructed_reshaped, output_dir, i)

            # Plot the reconstruction error
            plot_reconstruction_error(sample, reconstructed_reshaped, output_dir, i)
        else:
            # Fallback to original code
            plot_spectrograms(sample, reconstructed.squeeze(0), output_dir, i)
            plot_reconstruction_error(sample, reconstructed.squeeze(0), output_dir, i)

        # For bottleneck, also reshape if needed
        if bottleneck.dim() == 4:  # [time, batch, channels, freq]
            time, batch, channels, freq = bottleneck.shape
            # Reshape to [channels, freq, time]
            bottleneck_reshaped = bottleneck.permute(2, 3, 0, 1).reshape(channels, freq, time)
            plot_bottleneck(bottleneck_reshaped, output_dir, i)
        else:
            plot_bottleneck(bottleneck.squeeze(0), output_dir, i)

    typer.echo(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
