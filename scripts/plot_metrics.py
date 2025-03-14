import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_report(df, output_path):
    """
    Generate a detailed report of the metrics from the dataframe.

    Args:
        df: DataFrame containing metrics
        output_path: Path to save the report
    """
    with open(output_path, "w") as f:
        f.write("# Training Metrics Report\n\n")

        # List all available metrics
        all_metrics = [col for col in df.columns if col not in ["step", "epoch", "lr-Adam"]]
        f.write("## Available Metrics\n")
        f.write(f"{', '.join(all_metrics)}\n\n")

        # Calculate statistics for each metric
        for metric in all_metrics:
            # Skip if all values are NaN
            if df[metric].isna().all():
                continue

            metric_data = df[df[metric].notna()]
            if len(metric_data) == 0:
                continue

            f.write(f"## {metric}\n")
            f.write(f"Min: {metric_data[metric].min():.6f}\n")
            f.write(f"Max: {metric_data[metric].max():.6f}\n")
            f.write(f"Mean: {metric_data[metric].mean():.6f}\n")
            f.write(f"Final: {metric_data[metric].iloc[-1]:.6f}\n")

            if "step" in df.columns:
                min_step = metric_data.loc[metric_data[metric].idxmin(), "step"]
                max_step = metric_data.loc[metric_data[metric].idxmax(), "step"]
                f.write(f"Min at step: {min_step}\n")
                f.write(f"Max at step: {max_step}\n")

            f.write("\n")

        # Add some analysis on convergence if we have loss metrics
        if "train/loss" in df.columns and df["train/loss"].notna().any():
            train_loss = df[df["train/loss"].notna()]
            final_loss = train_loss["train/loss"].iloc[-1]
            final_step = train_loss["step"].iloc[-1]

            # Calculate step at which we reached 1.1 * final_loss
            threshold = 1.1 * final_loss
            steps_to_threshold = None

            for idx, row in train_loss.iterrows():
                if row["train/loss"] <= threshold:
                    steps_to_threshold = row["step"]
                    break

            f.write("## Convergence Analysis\n")
            f.write(f"Total training steps: {final_step}\n")

            if steps_to_threshold is not None:
                threshold_percent = (steps_to_threshold / final_step) * 100
                f.write(
                    f"Steps to reach 1.1 of final loss: {steps_to_threshold} "
                    f"({threshold_percent:.2f}% of training)\n"
                )
            else:
                f.write("Did not reach 1.1 of final loss during training\n")

            f.write("\n")


def plot_metrics(csv_file, output_dir=None):
    """
    Plot training and validation metrics from a CSV file.

    Args:
        csv_file: Path to the CSV file containing metrics
        output_dir: Directory to save plots (if None, will display instead)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create figures directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate report if output_dir is provided
    if output_dir:
        report_path = os.path.join(output_dir, "metrics_report.txt")
        generate_report(df, report_path)
        print(f"Report generated at: {report_path}")

    # Create figure for loss
    plt.figure(figsize=(12, 6))

    # Plot training loss if it exists
    if "train/loss" in df.columns:
        # Filter out rows where train/loss is NaN
        train_loss_df = df[df["train/loss"].notna()]
        plt.plot(train_loss_df["step"], train_loss_df["train/loss"], label="Training Loss")

    # Plot validation loss if it exists
    if "val/loss" in df.columns:
        # Filter out rows where val/loss is NaN
        val_loss_df = df[df["val/loss"].notna()]
        plt.plot(
            val_loss_df["step"], val_loss_df["val/loss"], label="Validation Loss", color="orange"
        )

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(os.path.join(output_dir, "loss_plot.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # Create figure for CER
    plt.figure(figsize=(12, 6))

    # Plot training CER if it exists
    if "train/CER" in df.columns:
        # Filter out rows where train/CER is NaN
        train_cer_df = df[df["train/CER"].notna()]
        plt.plot(train_cer_df["step"], train_cer_df["train/CER"], label="Training CER")

    # Plot validation CER if it exists
    if "val/CER" in df.columns:
        # Filter out rows where val/CER is NaN
        val_cer_df = df[df["val/CER"].notna()]
        plt.plot(val_cer_df["step"], val_cer_df["val/CER"], label="Validation CER", color="orange")

    plt.xlabel("Steps")
    plt.ylabel("Character Error Rate (CER)")
    plt.title("Training and Validation CER")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(os.path.join(output_dir, "cer_plot.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()

    # Optional: Plot more detailed view with lower CER values (excluding initial high values)
    if "val/CER" in df.columns or "train/CER" in df.columns:
        plt.figure(figsize=(12, 6))

        threshold = 200  # Maximum CER to include in zoomed plot

        if "train/CER" in df.columns:
            train_cer_df = df[df["train/CER"].notna()]
            zoomed_train = train_cer_df[train_cer_df["train/CER"] < threshold]
            if not zoomed_train.empty:
                plt.plot(zoomed_train["step"], zoomed_train["train/CER"], label="Training CER")

        if "val/CER" in df.columns:
            val_cer_df = df[df["val/CER"].notna()]
            zoomed_val = val_cer_df[val_cer_df["val/CER"] < threshold]
            if not zoomed_val.empty:
                plt.plot(
                    zoomed_val["step"],
                    zoomed_val["val/CER"],
                    label="Validation CER",
                    color="orange",
                )

        plt.xlabel("Steps")
        plt.ylabel("Character Error Rate (CER)")
        plt.title("Training and Validation CER (Zoomed View)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(
                os.path.join(output_dir, "cer_zoomed_plot.png"), dpi=300, bbox_inches="tight"
            )
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training and validation metrics from CSV files"
    )
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing metrics")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        help="Directory to save plots (optional)",
        dest="output_dir",
    )

    args = parser.parse_args()
    plot_metrics(args.csv_file, args.output_dir)
