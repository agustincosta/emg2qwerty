#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from emg2qwerty.data import WindowedEMGDataset
from emg2qwerty.lightning import AutoencoderModule
from emg2qwerty.transforms import SpectrogramTransform

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize autoencoder reconstructions")
    parser.add_argument(
        "--session", type=str, required=True, help="Path to EMG session file (HDF5)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained autoencoder checkpoint"
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load the autoencoder model
    log.info(f"Loading model from checkpoint: {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint, map_location=lambda storage, loc: storage, weights_only=False
    )

    # Determine model parameters from checkpoint
    state_dict = checkpoint["state_dict"]
    # Find keys with 'encoder.0.weight' to get in_channels
    encoder_keys = [k for k in state_dict.keys() if "encoder.0.weight" in k]
    if encoder_keys:
        in_channels = state_dict[encoder_keys[0]].shape[1]
    else:
        in_channels = 32  # Default (2 bands * 16 electrodes)

    # Find keys with encoder output to get bottleneck_channels
    bottleneck_keys = [k for k in state_dict.keys() if "encoder" in k and "weight" in k]
    bottleneck_keys.sort(key=lambda x: int(x.split(".")[1]) if x.split(".")[1].isdigit() else 0)
    if bottleneck_keys and len(bottleneck_keys) >= 2:
        bottleneck_channels = state_dict[bottleneck_keys[-2]].shape[0]
    else:
        bottleneck_channels = 16  # Default

    log.info(
        f"Model parameters: in_channels={in_channels}, bottleneck_channels={bottleneck_channels}"
    )

    # Create the model
    model = AutoencoderModule(
        in_channels=in_channels,
        bottleneck_channels=bottleneck_channels,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Check which session file we're using
    log.info(f"Loading data from session: {args.session}")

    # Initialize transform (similar to one used during training)
    transform = SpectrogramTransform(
        fs=2000,  # EMG sampling frequency (2kHz)
        window_size=256,
        hop_size=128,
        n_fft=512,
        standardize=True,
    )

    # Create dataset
    dataset = WindowedEMGDataset(
        hdf5_path=Path(args.session),
        window_length=2000,  # 1 second at 2kHz
        stride=2000,
        transform=transform,
    )

    # Create dataloader with batch size 1 to process one window at a time
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get samples for visualization
    samples_count = 0
    for batch_idx, (inputs, _) in enumerate(dataloader):
        if samples_count >= args.num_samples:
            break

        inputs = inputs.to(device)

        # Get reconstructions
        with torch.no_grad():
            _, reconstructed = model.autoencoder(inputs)

        # Convert to numpy for plotting
        inputs_np = inputs.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f"EMG Spectrogram Reconstruction - Sample {samples_count + 1}", fontsize=16)

        # Plot a subset of channels (4x4 grid)
        for i in range(4):
            for j in range(4):
                channel_idx = i * 4 + j
                band_idx = 0 if channel_idx < 8 else 1
                electr_idx = channel_idx % 8

                # Original spectrogram (left band, electrode i)
                ax = axes[i, j]
                im = ax.imshow(
                    inputs_np[0, 0, band_idx, electr_idx].T,  # Transpose for better visualization
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                )
                ax.set_title(f"Original B{band_idx}E{electr_idx}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Frequency")

                # Add a colorbar
                fig.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()
        plt.savefig(output_dir / f"original_spectrograms_{samples_count}.png")

        # Create a second figure for reconstructions
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f"EMG Spectrogram Reconstruction - Sample {samples_count + 1}", fontsize=16)

        # Plot the reconstructed spectrograms
        for i in range(4):
            for j in range(4):
                channel_idx = i * 4 + j
                band_idx = 0 if channel_idx < 8 else 1
                electr_idx = channel_idx % 8

                # Reconstructed spectrogram
                ax = axes[i, j]
                im = ax.imshow(
                    reconstructed_np[
                        0, 0, band_idx, electr_idx
                    ].T,  # Transpose for better visualization
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                )
                ax.set_title(f"Reconstructed B{band_idx}E{electr_idx}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Frequency")

                # Add a colorbar
                fig.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()
        plt.savefig(output_dir / f"reconstructed_spectrograms_{samples_count}.png")

        # Create a third figure for comparison (original vs reconstructed)
        for channel_idx in range(4):  # Just show 4 channels for detailed comparison
            band_idx = 0 if channel_idx < 2 else 1
            electr_idx = channel_idx % 2

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(
                f"Original vs Reconstructed - Sample {samples_count + 1}, B{band_idx}E{electr_idx}",
                fontsize=14,
            )

            # Original
            im1 = axes[0].imshow(
                inputs_np[0, 0, band_idx, electr_idx].T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
            )
            axes[0].set_title("Original")
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel("Frequency")
            fig.colorbar(im1, ax=axes[0])

            # Reconstructed
            im2 = axes[1].imshow(
                reconstructed_np[0, 0, band_idx, electr_idx].T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
            )
            axes[1].set_title("Reconstructed")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Frequency")
            fig.colorbar(im2, ax=axes[1])

            plt.tight_layout()
            plt.savefig(output_dir / f"comparison_{samples_count}_B{band_idx}E{electr_idx}.png")
            plt.close()

        samples_count += 1

    log.info(f"Saved {samples_count} sample visualizations to {output_dir}")


if __name__ == "__main__":
    main()
