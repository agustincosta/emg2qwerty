import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import typer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore


def find_tensorboard_logs(root_dir: str) -> List[str]:
    """Find all TensorBoard log directories in the given root directory and its subdirectories."""
    log_dirs = []

    # Walk through all subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        # Check if this directory contains TensorBoard event files
        if any(f.startswith("events.out.tfevents.") for f in filenames):
            log_dirs.append(dirpath)

    if not log_dirs:
        raise ValueError(f"No TensorBoard event files found in {root_dir} or its subdirectories")

    typer.echo(f"Found {len(log_dirs)} TensorBoard log directories")
    return log_dirs


def extract_tensorboard_data(log_dir: str) -> Dict:
    """Extract data from TensorBoard logs into a dictionary."""
    event_paths = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

    if not event_paths:
        typer.echo(f"No TensorBoard event files found in {log_dir}")
        return {}

    # Use the most recent event file
    event_path = sorted(event_paths)[-1]
    typer.echo(f"Reading TensorBoard logs from: {event_path}")

    event_acc = EventAccumulator(event_path)
    event_acc.Reload()

    # Get available tags (metrics)
    tags = event_acc.Tags()["scalars"]

    # Extract data for each tag
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = {
            "step": [event.step for event in events],
            "value": [event.value for event in events],
            "wall_time": [event.wall_time for event in events],
        }

    return data


def plot_losses(all_data, output_dir, show=True):
    """Plot training, validation, and test losses for all runs."""
    # Track which runs have which metrics for the details file
    run_details = {}

    # Use different line styles and colors for different runs
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    line_styles = ["-", "--", "-.", ":"]

    # Create separate plots for each metric type
    for metric_type in ["train/loss", "val/loss", "test/loss"]:
        plt.figure(figsize=(14, 10))
        legend_entries = []

        for i, (run_dir, data) in enumerate(all_data.items()):
            run_name = os.path.basename(os.path.normpath(run_dir))
            color_idx = i % len(colors)
            style_idx = (i // len(colors)) % len(line_styles)

            if run_name not in run_details:
                run_details[run_name] = {"directory": run_dir, "metrics": []}

            # Plot the specific metric if available
            if metric_type in data:
                metric_data = data[metric_type]
                plt.plot(
                    metric_data["step"],
                    metric_data["value"],
                    label=f"{run_name}",
                    color=colors[color_idx],
                    linestyle=line_styles[style_idx],
                )
                legend_entries.append(f"{run_name}")

                if metric_type not in run_details[run_name]["metrics"]:
                    run_details[run_name]["metrics"].append(metric_type)

                # Add min/max/final values to details
                run_details[run_name][f"{metric_type}_min"] = min(metric_data["value"])
                run_details[run_name][f"{metric_type}_max"] = max(metric_data["value"])
                run_details[run_name][f"{metric_type}_final"] = metric_data["value"][-1]

                # Calculate additional metrics for training data
                if metric_type == "train/loss":
                    # Calculate total steps
                    total_steps = metric_data["step"][-1]
                    run_details[run_name]["total_steps"] = total_steps

                    # Calculate steps to reach threshold (1.1 of final loss)
                    threshold = 1.1 * metric_data["value"][-1]
                    steps_to_threshold = None
                    for idx, value in enumerate(metric_data["value"]):
                        if value <= threshold:
                            steps_to_threshold = metric_data["step"][idx]
                            break
                    run_details[run_name]["steps_to_threshold"] = steps_to_threshold

        if legend_entries:  # Only save if there's data to plot
            plt.xlabel("Step")
            plt.ylabel(metric_type.split("/")[1].capitalize())
            plt.title(f"{metric_type.split('/')[1].capitalize()} Across All Runs")
            plt.legend(legend_entries)
            plt.grid(True, linestyle="--", alpha=0.7)

            # Save the plot
            os.makedirs(output_dir, exist_ok=True)
            metric_name = metric_type.replace("/", "_")
            output_path = os.path.join(output_dir, f"{metric_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            typer.echo(f"{metric_type} plot saved to {output_path}")

            if show:
                plt.show()
            plt.close()

    # Create separate plots for CER metrics
    for metric_type in ["train/CER", "val/CER", "test/CER"]:
        plt.figure(figsize=(14, 10))
        legend_entries = []

        for i, (run_dir, data) in enumerate(all_data.items()):
            run_name = os.path.basename(os.path.normpath(run_dir))
            color_idx = i % len(colors)
            style_idx = (i // len(colors)) % len(line_styles)

            if metric_type in data:
                cer_data = data[metric_type]
                plt.plot(
                    cer_data["step"],
                    cer_data["value"],
                    label=f"{run_name}",
                    color=colors[color_idx],
                    linestyle=line_styles[style_idx],
                )
                legend_entries.append(f"{run_name}")

                if metric_type not in run_details[run_name]["metrics"]:
                    run_details[run_name]["metrics"].append(metric_type)

                # Add min/max/final CER to details
                run_details[run_name][f"{metric_type}_min"] = min(cer_data["value"])
                run_details[run_name][f"{metric_type}_max"] = max(cer_data["value"])
                run_details[run_name][f"{metric_type}_final"] = cer_data["value"][-1]

        if legend_entries:  # Only save if there's data to plot
            plt.xlabel("Step")
            plt.ylabel("Character Error Rate (%)")
            plt.title(f"{metric_type.split('/')[0].capitalize()} Character Error Rate")
            plt.legend(legend_entries)
            plt.grid(True, linestyle="--", alpha=0.7)

            # Save the plot
            metric_name = metric_type.replace("/", "_")
            output_path = os.path.join(output_dir, f"{metric_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            typer.echo(f"{metric_type} plot saved to {output_path}")

            if show:
                plt.show()
            plt.close()

    # Save details to text file
    details_path = os.path.join(output_dir, "run_details.txt")
    with open(details_path, "w") as f:
        f.write("# Training Run Details\n\n")
        for run_name, details in run_details.items():
            f.write(f"## {run_name}\n")
            f.write(f"Directory: {details['directory']}\n")
            f.write(f"Available metrics: {', '.join(details['metrics'])}\n")

            if "total_steps" in details:
                f.write(f"Total training steps: {details['total_steps']}\n")

                if "steps_to_threshold" in details and details["steps_to_threshold"] is not None:
                    threshold_percent = (
                        details["steps_to_threshold"] / details["total_steps"]
                    ) * 100
                    f.write(
                        f"Steps to reach 1.1 of final loss: {details['steps_to_threshold']}"
                        f" ({threshold_percent:.2f}% of training)\n"
                    )
                else:
                    f.write("Did not reach 1.1 of final loss during training\n")

            f.write("\n")

            if "train/loss" in details["metrics"]:
                f.write("### Training Loss\n")
                f.write(f"Min: {details['train/loss_min']:.6f}\n")
                f.write(f"Max: {details['train/loss_max']:.6f}\n")
                f.write(f"Final: {details['train/loss_final']:.6f}\n\n")

            if "val/loss" in details["metrics"]:
                f.write("### Validation Loss\n")
                f.write(f"Min: {details['val/loss_min']:.6f}\n")
                f.write(f"Max: {details['val/loss_max']:.6f}\n")
                f.write(f"Final: {details['val/loss_final']:.6f}\n\n")

            if "test/loss" in details["metrics"]:
                f.write("### Test Loss\n")
                f.write(f"Min: {details['test/loss_min']:.6f}\n")
                f.write(f"Max: {details['test/loss_max']:.6f}\n")
                f.write(f"Final: {details['test/loss_final']:.6f}\n\n")

            # Add CER metrics to details
            if "train/CER" in details["metrics"]:
                f.write("### Training CER\n")
                f.write(f"Min: {details['train/CER_min']:.6f}\n")
                f.write(f"Max: {details['train/CER_max']:.6f}\n")
                f.write(f"Final: {details['train/CER_final']:.6f}\n\n")

            if "val/CER" in details["metrics"]:
                f.write("### Validation CER\n")
                f.write(f"Min: {details['val/CER_min']:.6f}\n")
                f.write(f"Max: {details['val/CER_max']:.6f}\n")
                f.write(f"Final: {details['val/CER_final']:.6f}\n\n")

            if "test/CER" in details["metrics"]:
                f.write("### Test CER\n")
                f.write(f"Min: {details['test/CER_min']:.6f}\n")
                f.write(f"Max: {details['test/CER_max']:.6f}\n")
                f.write(f"Final: {details['test/CER_final']:.6f}\n\n")

            f.write("-" * 50 + "\n\n")

    typer.echo(f"Run details saved to {details_path}")


def plot_cer_metrics(all_data, output_dir, show=False):
    """Plot CER metrics for all runs."""
    plt.figure(figsize=(14, 10))

    # Use different line styles and colors for different runs
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    line_styles = ["-", "--", "-.", ":"]

    legend_entries = []

    for i, (run_dir, data) in enumerate(all_data.items()):
        run_name = os.path.basename(os.path.normpath(run_dir))
        color_idx = i % len(colors)
        style_idx = (i // len(colors)) % len(line_styles)

        # Plot training CER
        if "train/CER" in data:
            train_data = data["train/CER"]
            plt.plot(
                train_data["step"],
                train_data["value"],
                label=f"{run_name} - Training CER",
                color=colors[color_idx],
                linestyle=line_styles[style_idx],
            )
            legend_entries.append(f"{run_name} - Training CER")

        # Plot validation CER
        if "val/CER" in data:
            val_data = data["val/CER"]
            plt.plot(
                val_data["step"],
                val_data["value"],
                label=f"{run_name} - Validation CER",
                color=colors[color_idx],
                linestyle=line_styles[(style_idx + 1) % len(line_styles)],
            )
            legend_entries.append(f"{run_name} - Validation CER")

        # Plot test CER if available
        if "test/CER" in data:
            test_data = data["test/CER"]
            plt.plot(
                test_data["step"],
                test_data["value"],
                label=f"{run_name} - Test CER",
                color=colors[color_idx],
                linestyle=line_styles[(style_idx + 2) % len(line_styles)],
            )
            legend_entries.append(f"{run_name} - Test CER")

    plt.xlabel("Step")
    plt.ylabel("Character Error Rate (CER)")
    plt.title("CER Metrics Across All Runs")
    plt.legend(legend_entries)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_cer_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    typer.echo(f"Combined CER metrics plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_individual_runs(all_data, output_dir, show=False):
    """Plot losses and CER metrics for each individual run separately."""
    for run_dir, data in all_data.items():
        run_name = os.path.basename(os.path.normpath(run_dir))

        # Plot losses
        plt.figure(figsize=(12, 8))

        # Plot training loss
        if "train/loss" in data:
            train_data = data["train/loss"]
            plt.plot(train_data["step"], train_data["value"], label="Training Loss", color="blue")

        # Plot validation loss
        if "val/loss" in data:
            val_data = data["val/loss"]
            plt.plot(val_data["step"], val_data["value"], label="Validation Loss", color="orange")

        # Plot test loss if available
        if "test/loss" in data:
            test_data = data["test/loss"]
            plt.plot(test_data["step"], test_data["value"], label="Test Loss", color="green")

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Losses for Run: {run_name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.ylim(0, 10)

        # Save the loss plot
        run_output_dir = os.path.join(output_dir, "individual_runs")
        os.makedirs(run_output_dir, exist_ok=True)
        output_path = os.path.join(run_output_dir, f"{run_name}_losses.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        typer.echo(f"Individual loss plot for {run_name} saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        # Plot CER metrics if available
        has_cer = any(metric in data for metric in ["train/CER", "val/CER", "test/CER"])
        if has_cer:
            plt.figure(figsize=(12, 8))

            # Plot training CER
            if "train/CER" in data:
                train_data = data["train/CER"]
                plt.plot(
                    train_data["step"], train_data["value"], label="Training CER", color="blue"
                )

            # Plot validation CER
            if "val/CER" in data:
                val_data = data["val/CER"]
                plt.plot(
                    val_data["step"], val_data["value"], label="Validation CER", color="orange"
                )

            # Plot test CER if available
            if "test/CER" in data:
                test_data = data["test/CER"]
                plt.plot(test_data["step"], test_data["value"], label="Test CER", color="green")

            plt.xlabel("Step")
            plt.ylabel("Character Error Rate (CER)")
            plt.title(f"CER Metrics for Run: {run_name}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)

            plt.ylim(-2, 102)  # Set y-axis limits to -2 to 102

            # Save the CER plot
            output_path = os.path.join(run_output_dir, f"{run_name}_cer.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            typer.echo(f"Individual CER plot for {run_name} saved to {output_path}")

            if show:
                plt.show()
            else:
                plt.close()


def main(
    log_dir: str = typer.Argument(
        ..., help="Directory containing TensorBoard logs and subdirectories"
    ),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory for plots"),
    no_show: bool = typer.Option(False, "--no-show", help="Do not display plots"),
    individual: bool = typer.Option(
        True, "--individual/--no-individual", help="Create individual plots for each run"
    ),
):
    """
    Plot losses and CER metrics from all TensorBoard logs in a directory and its subdirectories.

    This tool extracts data from TensorBoard event files and creates visualizations
    of training, validation, and test losses and CER metrics over time for all runs.
    """
    # Find all TensorBoard log directories
    log_dirs = find_tensorboard_logs(log_dir)

    # Extract data from each log directory
    all_data = {}
    for dir_path in log_dirs:
        data = extract_tensorboard_data(dir_path)
        if data:  # Only include directories with valid data
            all_data[dir_path] = data

    if not all_data:
        typer.echo("No valid TensorBoard data found in any directory.")
        return

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Plot combined losses from all runs
    plot_losses(all_data, output_dir=output, show=not no_show)

    # Plot combined CER metrics from all runs
    has_cer = any(
        any(metric in data for metric in ["train/CER", "val/CER", "test/CER"])
        for data in all_data.values()
    )
    if has_cer:
        plot_cer_metrics(all_data, output_dir=output, show=not no_show)

    # Plot individual runs if requested
    if individual:
        plot_individual_runs(all_data, output_dir=output, show=not no_show)


if __name__ == "__main__":
    typer.run(main)
