"""Visualization scripts for training metrics and analysis.

This script loads training logs and checkpoint data to produce
comprehensive visualizations of the training process.

Usage
-----
    # Visualize training metrics from default experiment directory
    python visualize.py

    # Visualize a specific experiment
    python visualize.py --experiment-dir experiments/run_001

    # Visualize multiple experiments together
    python visualize.py --experiment-dir experiments/run_001 experiments/run_002

    # Save plots to file instead of displaying
    python visualize.py --output-dir plots/

    # Visualize only specific metrics
    python visualize.py --metrics loss epsilon reward

    # Custom font and figure size
    python visualize.py --font-size 12 --figure-size 12,8

Key features
------------
- Training loss curve with moving average
- TD error over time
- Epsilon decay schedule
- Replay buffer size evolution
- Episode reward distribution
- Multiple experiment comparison
- Auto-scaling and formatting
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import matplotlib, provide helpful error if not available
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for servers
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
    print("Proceeding with text-based output only.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _legend_if_labeled(ax: "plt.Axes", **kwargs: Any) -> None:
    """Call legend only when at least one artist has a label."""
    if not HAS_MATPLOTLIB:
        return
    _handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def _hydrate_from_episode_metrics(data: Dict[str, Any]) -> None:
    """Map Path B ``episode_metrics.json`` into legacy series when TB/CSV absent."""
    episodes = data.get("episode_metrics") or []
    if not episodes:
        return

    if not data["loss_history"]:
        data["loss_history"] = [
            (int(e.get("episode", i + 1)), float(e.get("avg_loss", 0.0)))
            for i, e in enumerate(episodes)
        ]
    if not data["epsilon_history"]:
        data["epsilon_history"] = [
            (int(e.get("episode", i + 1)), float(e.get("epsilon", 0.0)))
            for i, e in enumerate(episodes)
        ]
    if not data["buffer_size_history"]:
        data["buffer_size_history"] = [
            (int(e.get("episode", i + 1)), int(e.get("buffer_size", 0)))
            for i, e in enumerate(episodes)
        ]
    if not data["eval_metrics"]:
        data["eval_metrics"] = [
            {
                "step": int(e.get("episode", i + 1)),
                "mean_reward": float(e.get("shaped_reward", 0.0)),
                "std_reward": 0.0,
                "min_reward": float(e.get("raw_reward", 0.0)),
                "max_reward": float(e.get("shaped_reward", 0.0)),
            }
            for i, e in enumerate(episodes)
        ]
    if not data["final_eval"]:
        shaped = [float(e.get("shaped_reward", 0.0)) for e in episodes]
        data["final_eval"] = {
            "mean_reward": float(np.mean(shaped)),
            "std_reward": float(np.std(shaped)),
            "final_episode": int(episodes[-1].get("episode", len(episodes))),
        }


# ──────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────

class TrainingMetricsLoader:
    """Load and process training metrics from experiment directories.

    Parameters
    ----------
    experiment_dirs : list of str or Path
        Paths to experiment directories containing training logs.
    """

    def __init__(self, experiment_dirs: List[str]):
        self.experiment_dirs = [Path(d) for d in experiment_dirs]
        self.metrics = {}

    def load_all(self) -> None:
        """Load metrics from all experiment directories."""
        for exp_dir in self.experiment_dirs:
            exp_name = exp_dir.name
            self.metrics[exp_name] = self._load_experiment(exp_dir)

    def _load_experiment(self, exp_dir: Path) -> Dict[str, Any]:
        """Load metrics from a single experiment directory."""
        data = {
            "path": exp_dir,
            "name": exp_dir.name,
            "config": {},
            "loss_history": [],
            "epsilon_history": [],
            "buffer_size_history": [],
            "checkpoint_metrics": [],
            "eval_metrics": [],
            "final_eval": {},
            "training_timestamps": [],
        }

        # Load config
        config_path = exp_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data["config"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        # Load training logs (if tensorboard logs exist)
        tb_dir = exp_dir / "logs" / "tensorboard"
        if tb_dir.exists():
            data["loss_history"] = self._parse_tensorboard_logs(
                tb_dir, "train/loss"
            )
            data["epsilon_history"] = self._parse_tensorboard_logs(
                tb_dir, "train/epsilon"
            )
            data["buffer_size_history"] = self._parse_tensorboard_logs(
                tb_dir, "train/buffer_size"
            )

        # Load eval metrics CSV
        eval_csv = exp_dir / "metrics" / "eval_metrics.csv"
        if eval_csv.exists():
            data["eval_metrics"] = self._load_eval_csv(eval_csv)

        # Load Path B / self-play episode metrics
        episode_metrics = exp_dir / "metrics" / "episode_metrics.json"
        if episode_metrics.exists():
            try:
                with open(episode_metrics) as f:
                    ep_data = json.load(f)
                    data["episode_metrics"] = ep_data.get("episodes", [])
            except Exception as e:
                logger.warning(f"Failed to load episode metrics: {e}")

        win_rate_eval = exp_dir / "metrics" / "win_rate_eval.json"
        if win_rate_eval.exists():
            try:
                with open(win_rate_eval) as f:
                    data["win_rate_eval"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load win rate eval: {e}")

        final_eval = exp_dir / "metrics" / "final_eval.json"
        if final_eval.exists():
            try:
                with open(final_eval) as f:
                    data["final_eval"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load final eval: {e}")

        # Load checkpoint metadata
        checkpoints_dir = exp_dir / "checkpoints"
        if checkpoints_dir.exists():
            for meta_file in sorted(checkpoints_dir.glob("metadata_*.json")):
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                        data["checkpoint_metrics"].append(meta)
                except Exception:
                    continue

        _hydrate_from_episode_metrics(data)
        return data

    def _parse_tensorboard_logs(
        self, tb_dir: Path, tag: str
    ) -> List[Tuple[int, float]]:
        """Parse scalar logs from TensorBoard directory.

        This is a simplified parser for TensorBoard's binary format.
        For production use, consider using tensorboard module directly.
        """
        # Try to find event files
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return []

        # Simple text-based log parsing
        logs = []
        for event_file in event_files[:1]:  # Use first event file
            try:
                with open(event_file, "rb") as f:
                    content = f.read()
                    # Look for the tag pattern in the binary content
                    tag_bytes = tag.encode("utf-8")
                    if tag_bytes in content:
                        # Extract values (simplified)
                        pos = content.find(tag_bytes)
                        if pos >= 0:
                            # This is a rough extraction - for full parsing,
                            # use tensorboard module
                            pass
            except Exception:
                continue

        return logs

    def _load_eval_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load evaluation metrics from CSV file."""
        import csv

        metrics = []
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        metrics.append({
                            "step": int(row.get("timestep", 0)),
                            "mean_reward": float(row.get("mean_reward", 0)),
                            "std_reward": float(row.get("std_reward", 0)),
                            "min_reward": float(row.get("min_reward", 0)),
                            "max_reward": float(row.get("max_reward", 0)),
                        })
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to load eval CSV: {e}")

        return metrics

    def get_loss_history(self, exp_name: str) -> List[Tuple[int, float]]:
        """Get loss history for an experiment.

        Returns
        -------
        history : list of (step, loss) tuples
        """
        return self.metrics.get(exp_name, {}).get("loss_history", [])

    def get_epsilon_history(self, exp_name: str) -> List[Tuple[int, float]]:
        """Get epsilon history for an experiment.

        Returns
        -------
        history : list of (step, epsilon) tuples
        """
        return self.metrics.get(exp_name, {}).get("epsilon_history", [])

    def get_buffer_size_history(self, exp_name: str) -> List[Tuple[int, int]]:
        """Get buffer size history for an experiment.

        Returns
        -------
        history : list of (step, buffer_size) tuples
        """
        return self.metrics.get(exp_name, {}).get("buffer_size_history", [])

    def get_eval_metrics(self, exp_name: str) -> List[Dict[str, Any]]:
        """Get evaluation metrics for an experiment.

        Returns
        -------
        metrics : list of dicts with step, mean_reward, etc.
        """
        return self.metrics.get(exp_name, {}).get("eval_metrics", [])

    def get_final_eval(self, exp_name: str) -> Dict[str, Any]:
        """Get final evaluation metrics for an experiment.

        Returns
        -------
        metrics : dict with mean_reward, std_reward, etc.
        """
        return self.metrics.get(exp_name, {}).get("final_eval", {})


# ──────────────────────────────────────────────────────────────
#  Visualization Functions
# ──────────────────────────────────────────────────────────────

class MetricsVisualizer:
    """Create comprehensive training metric visualizations.

    Parameters
    ----------
    font_size : int
        Font size for plots (default: 12).
    figure_size : tuple
        Figure size in inches (default: (12, 8)).
    output_dir : str or Path
        Directory to save plots.
    """

    def __init__(
        self,
        font_size: int = 12,
        figure_size: Tuple[int, int] = (12, 8),
        output_dir: Optional[str] = None,
    ):
        self.font_size = font_size
        self.figure_size = figure_size
        self.output_dir = Path(output_dir) if output_dir else None

        # Apply settings
        plt.rcParams.update({
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.titlesize": font_size + 4,
        })

    def plot_loss_curves(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
        window_size: int = 100,
    ) -> plt.Figure:
        """Plot training loss curves.

        Parameters
        ----------
        loader : TrainingMetricsLoader
            Loaded metrics.
        exp_names : list of str or None
            Experiment names to plot. If None, plot all.
        window_size : int
            Window size for moving average.

        Returns
        -------
        fig : matplotlib Figure
        """
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        fig, ax = plt.subplots(figsize=self.figure_size)
        plotted = False

        for exp_name in exp_names:
            if exp_name not in loader.metrics:
                continue

            loss_history = loader.get_loss_history(exp_name)
            if not loss_history:
                continue

            steps = [h[0] for h in loss_history]
            losses = [h[1] for h in loss_history]

            # Plot raw loss
            ax.plot(
                steps, losses, alpha=0.3, label=f"{exp_name} (raw)",
                linewidth=0.5, color="blue"
            )
            plotted = True

            # Plot moving average (adapt window for short self-play runs)
            ma_window = min(window_size, max(1, len(losses) // 3))
            if len(losses) > ma_window:
                ma = self._moving_average(losses, ma_window)
                ax.plot(
                    steps[ma_window - 1:], ma,
                    label=f"{exp_name} (MA{ma_window})",
                    linewidth=2, color="blue"
                )

        if not plotted:
            plt.close(fig)
            return None

        ax.set_title("Training Loss")
        ax.set_xlabel("Episode" if max(steps, default=0) <= 1000 else "Step")
        ax.set_ylabel("Loss")
        _legend_if_labeled(ax)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("symlog")  # Log scale for better visibility

        return fig

    def plot_epsilon_decay(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Plot epsilon decay schedules.

        Parameters
        ----------
        loader : TrainingMetricsLoader
            Loaded metrics.
        exp_names : list of str or None
            Experiment names to plot.

        Returns
        -------
        fig : matplotlib Figure
        """
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        fig, ax = plt.subplots(figsize=self.figure_size)
        plotted = False

        for exp_name in exp_names:
            if exp_name not in loader.metrics:
                continue

            epsilon_history = loader.get_epsilon_history(exp_name)
            if not epsilon_history:
                continue

            steps = [h[0] for h in epsilon_history]
            epsilons = [h[1] for h in epsilon_history]

            ax.plot(
                steps, epsilons,
                label=f"{exp_name}", linewidth=2
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return None

        ax.set_title("Epsilon Decay Schedule")
        ax.set_xlabel("Episode" if max(steps, default=0) <= 1000 else "Step")
        ax.set_ylabel("Epsilon")
        _legend_if_labeled(ax)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        return fig

    def plot_buffer_size(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Plot replay buffer size over time.

        Parameters
        ----------
        loader : TrainingMetricsLoader
            Loaded metrics.
        exp_names : list of str or None
            Experiment names to plot.

        Returns
        -------
        fig : matplotlib Figure
        """
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        fig, ax = plt.subplots(figsize=self.figure_size)
        plotted = False

        for exp_name in exp_names:
            if exp_name not in loader.metrics:
                continue

            buffer_history = loader.get_buffer_size_history(exp_name)
            if not buffer_history:
                continue

            steps = [h[0] for h in buffer_history]
            sizes = [h[1] for h in buffer_history]

            ax.plot(
                steps, sizes,
                label=f"{exp_name}", linewidth=2
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return None

        ax.set_title("Replay Buffer Size")
        ax.set_xlabel("Episode" if max(steps, default=0) <= 1000 else "Step")
        ax.set_ylabel("Buffer Size")
        _legend_if_labeled(ax)
        ax.grid(True, alpha=0.3)

        return fig

    def plot_eval_rewards(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
        window_size: int = 5,
    ) -> plt.Figure:
        """Plot evaluation reward metrics over time.

        Parameters
        ----------
        loader : TrainingMetricsLoader
            Loaded metrics.
        exp_names : list of str or None
            Experiment names to plot.
        window_size : int
            Window size for smoothing.

        Returns
        -------
        fig : matplotlib Figure
        """
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        fig, ax = plt.subplots(figsize=self.figure_size)
        plotted = False

        for exp_name in exp_names:
            if exp_name not in loader.metrics:
                continue

            eval_metrics = loader.get_eval_metrics(exp_name)
            if not eval_metrics:
                continue

            steps = [m["step"] for m in eval_metrics]
            means = [m["mean_reward"] for m in eval_metrics]

            # Plot mean reward
            ax.plot(
                steps, means,
                label=f"{exp_name} (shaped)", linewidth=2
            )
            plotted = True

            # Plot +/- std when available
            stds = [m.get("std_reward", 0) for m in eval_metrics]
            if any(s > 0 for s in stds):
                ax.fill_between(
                    steps,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="blue"
                )

        if not plotted:
            plt.close(fig)
            return None

        ax.set_title("Shaped Reward Over Episodes")
        ax.set_xlabel("Episode" if max(steps, default=0) <= 1000 else "Step")
        ax.set_ylabel("Mean Reward")
        _legend_if_labeled(ax)
        ax.grid(True, alpha=0.3)

        return fig

    def plot_final_reward_distribution(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Plot final evaluation reward distributions.

        Parameters
        ----------
        loader : TrainingMetricsLoader
            Loaded metrics.
        exp_names : list of str or None
            Experiment names to plot.

        Returns
        -------
        fig : matplotlib Figure
        """
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        fig, axes = plt.subplots(1, 2, figsize=(self.figure_size[0] * 2, self.figure_size[1]))

        # Bar chart of final metrics
        names = []
        mean_rewards = []
        std_rewards = []
        for exp_name in exp_names:
            final_eval = loader.get_final_eval(exp_name)
            if final_eval:
                names.append(exp_name)
                mean_rewards.append(final_eval.get("mean_reward", 0))
                std_rewards.append(final_eval.get("std_reward", 0))

        if names:
            x = np.arange(len(names))
            width = 0.35
            axes[0].bar(x - width/2, mean_rewards, width, label="Mean", yerr=std_rewards, capsize=5)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(names, rotation=45, ha="right")
            axes[0].set_title("Final Evaluation Rewards")
            axes[0].set_ylabel("Mean Reward")
            _legend_if_labeled(axes[0])
            axes[0].grid(True, alpha=0.3, axis="y")
        else:
            plt.close(fig)
            return None

        # Summary statistics table
        axes[1].axis("off")
        table_data = []
        for i, exp_name in enumerate(names):
            final_eval = loader.get_final_eval(exp_name)
            if final_eval:
                table_data.append([
                    exp_name,
                    f"{final_eval.get('mean_reward', 0):.2f}",
                    f"{final_eval.get('std_reward', 0):.2f}",
                    f"{final_eval.get('min_reward', 0):.2f}",
                    f"{final_eval.get('max_reward', 0):.2f}",
                ])

        if table_data:
            columns = ["Experiment", "Mean", "Std", "Min", "Max"]
            table = axes[1].table(
                cellText=table_data,
                colLabels=columns,
                loc="center",
                cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(self.font_size - 2)
            table.scale(1, 2)
            axes[1].set_title("Final Evaluation Summary")

        plt.tight_layout()
        return fig

    def plot_self_play_episode_metrics(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
    ) -> Optional[plt.Figure]:
        """Plot Path B self-play episode metrics (epsilon, reward, loss)."""
        if not HAS_MATPLOTLIB:
            return None
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        has_data = any(
            loader.metrics.get(name, {}).get("episode_metrics")
            for name in exp_names
        )
        if not has_data:
            return None

        fig, axes = plt.subplots(3, 1, figsize=self.figure_size, sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(exp_names), 1)))

        for idx, exp_name in enumerate(exp_names):
            episodes = loader.metrics.get(exp_name, {}).get("episode_metrics", [])
            if not episodes:
                continue
            xs = [e.get("episode", i + 1) for i, e in enumerate(episodes)]
            color = colors[idx % len(colors)]
            axes[0].plot(xs, [e.get("epsilon", 0) for e in episodes],
                         label=exp_name, color=color, linewidth=2)
            axes[1].plot(xs, [e.get("shaped_reward", 0) for e in episodes],
                         label=exp_name, color=color, linewidth=2)
            axes[2].plot(xs, [e.get("avg_loss", 0) for e in episodes],
                         label=exp_name, color=color, linewidth=2)

        axes[0].set_ylabel("Epsilon")
        axes[0].set_title("Self-Play Episode Metrics")
        _legend_if_labeled(axes[0], fontsize=self.font_size - 2)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("Shaped Reward")
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel("Avg Loss")
        axes[2].set_xlabel("Episode")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_win_rate_eval(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
    ) -> Optional[plt.Figure]:
        """Plot rubric-aligned win-rate evaluation vs baseline."""
        if not HAS_MATPLOTLIB:
            return None
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        rows = []
        for exp_name in exp_names:
            wr = loader.metrics.get(exp_name, {}).get("win_rate_eval")
            if wr:
                rows.append((exp_name, wr))

        if not rows:
            return None

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 2), self.figure_size[1] * 0.6))
        labels = [r[0] for r in rows]
        win_rates = [float(r[1].get("win_rate", 0)) for r in rows]
        tie_rates = [float(r[1].get("tie_rate", 0)) for r in rows]
        loss_rates = [float(r[1].get("loss_rate", 0)) for r in rows]
        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, win_rates, width, label="Win", color="#2ecc71")
        ax.bar(x, tie_rates, width, label="Tie", color="#f1c40f")
        ax.bar(x + width, loss_rates, width, label="Loss", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.set_title("Win Rate Eval vs Baseline (Rubric)")
        _legend_if_labeled(ax)
        ax.grid(True, axis="y", alpha=0.3)

        for i, wr in enumerate(win_rates):
            ax.text(i - width, win_rates[i] + 0.02, f"{win_rates[i]:.0%}",
                    ha="center", fontsize=self.font_size - 2)

        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        loader: TrainingMetricsLoader,
        exp_names: Optional[List[str]] = None,
    ) -> List[plt.Figure]:
        """Create a comprehensive comparison of all experiments.

        Parameters
        ----------
        loader : TrainingMetricsLoader
            Loaded metrics.
        exp_names : list of str or None
            Experiment names to compare.

        Returns
        -------
        figures : list of matplotlib Figures
        """
        if exp_names is None:
            exp_names = list(loader.metrics.keys())

        candidates = [
            self.plot_loss_curves(loader, exp_names),
            self.plot_epsilon_decay(loader, exp_names),
            self.plot_buffer_size(loader, exp_names),
            self.plot_eval_rewards(loader, exp_names),
            self.plot_final_reward_distribution(loader, exp_names),
            self.plot_self_play_episode_metrics(loader, exp_names),
            self.plot_win_rate_eval(loader, exp_names),
        ]
        figures = [fig for fig in candidates if fig is not None]
        return figures

    def print_summary(self, loader: TrainingMetricsLoader) -> None:
        """Print a text summary of training results.

        Parameters
        ----------
        loader : TrainingMetricsLoader
            Loaded metrics.
        """
        print("\n" + "=" * 70)
        print("TRAINING METRICS SUMMARY")
        print("=" * 70)

        for exp_name, data in loader.metrics.items():
            print(f"\nExperiment: {exp_name}")
            print("-" * 50)

            # Training stats
            config = data.get("config", {})
            print(f"  Config: {json.dumps(config, indent=8)}")

            # Loss stats
            loss_history = data.get("loss_history", [])
            if loss_history:
                losses = [h[1] for h in loss_history]
                print(f"  Loss: mean={np.mean(losses):.4f}, "
                      f"min={np.min(losses):.4f}, "
                      f"max={np.max(losses):.4f}")

            # Epsilon stats
            epsilon_history = data.get("epsilon_history", [])
            if epsilon_history:
                epsilons = [h[1] for h in epsilon_history]
                print(f"  Epsilon: final={epsilons[-1]:.4f}")

            # Buffer stats
            buffer_history = data.get("buffer_size_history", [])
            if buffer_history:
                sizes = [h[1] for h in buffer_history]
                print(f"  Buffer: final={sizes[-1]}, max={max(sizes)}")

            # Eval stats
            eval_metrics = data.get("eval_metrics", [])
            if eval_metrics:
                means = [m["mean_reward"] for m in eval_metrics]
                print(f"  Eval rewards: mean={np.mean(means):.2f}")

            # Final eval
            final_eval = data.get("final_eval", {})
            if final_eval:
                print(f"  Final eval: {json.dumps(final_eval, indent=8)}")

            episode_metrics = data.get("episode_metrics", [])
            if episode_metrics:
                shaped = [e.get("shaped_reward", 0) for e in episode_metrics]
                losses = [e.get("avg_loss", 0) for e in episode_metrics]
                print(f"  Self-play episodes: {len(episode_metrics)}")
                print(f"  Shaped reward: mean={np.mean(shaped):.2f}, final={shaped[-1]:.2f}")
                print(f"  Episode avg loss: mean={np.mean(losses):.4f}, final={losses[-1]:.4f}")

            win_rate_eval = data.get("win_rate_eval", {})
            if win_rate_eval:
                print(
                    f"  Win rate eval: win={win_rate_eval.get('win_rate', 0):.1%}, "
                    f"tie={win_rate_eval.get('tie_rate', 0):.1%}, "
                    f"loss={win_rate_eval.get('loss_rate', 0):.1%} "
                    f"(n={win_rate_eval.get('n_episodes', '?')})"
                )

        print("\n" + "=" * 70)

    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Calculate moving average.

        Parameters
        ----------
        data : list of float
            Input data.
        window : int
            Window size.

        Returns
        -------
        ma : numpy array
            Moving average values.
        """
        if len(data) < window:
            return np.array(data)

        ma = np.convolve(data, np.ones(window)/window, mode="valid")
        return ma

    def save_figures(
        self,
        figures: List[plt.Figure],
        exp_names: List[str],
        *,
        close: bool = True,
    ) -> None:
        """Save all figures to files.

        Parameters
        ----------
        figures : list of Figure
            Figures to save.
        exp_names : list of str
            Experiment names for naming.
        close : bool
            Close figures after saving (set False to keep displaying in notebooks).
        """
        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig_names = [
            "training_loss",
            "epsilon_decay",
            "buffer_size",
            "eval_rewards",
            "final_rewards",
            "self_play_episodes",
            "win_rate_eval",
        ]

        for fig, name in zip(figures, fig_names[: len(figures)]):
            if exp_names and len(exp_names) == 1:
                filename = f"{exp_names[0]}_{name}.png"
            else:
                filename = f"{name}_all_experiments.png"

            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            if close:
                plt.close(fig)
            logger.info(f"Saved: {filepath}")

    def show_figures(self, figures: List[plt.Figure]) -> None:
        """Display all figures interactively.

        Parameters
        ----------
        figures : list of Figure
            Figures to display.
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available for display.")
            return

        for fig in figures:
            plt.figure(fig)
        plt.show()


# ──────────────────────────────────────────────────────────────
#  Main Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    """Main entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize training metrics for Kaggriculture DQN"
    )

    # Experiment directories
    parser.add_argument(
        "--experiment-dir",
        nargs="+",
        default=["experiments"],
        help="Experiment directory (or directories) to load metrics from",
    )

    # Plot options
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["loss", "epsilon", "reward", "buffer", "self_play", "win_rate", "all"],
        default=["all"],
        help="Which metrics to plot",
    )

    # Display options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots to",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (requires matplotlib)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Font size for plots",
    )
    parser.add_argument(
        "--figure-size",
        type=int,
        nargs=2,
        default=[12, 8],
        help="Figure size in inches (width height)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print text summary of metrics",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save plots to files",
    )

    args = parser.parse_args()

    # Check matplotlib
    if not HAS_MATPLOTLIB:
        if not args.no_save and args.output_dir:
            print("ERROR: matplotlib is required to save plots. Install with: pip install matplotlib")
            return
        if args.show:
            print("ERROR: matplotlib is required to display plots. Install with: pip install matplotlib")
            return
        print("WARNING: matplotlib not installed. Showing text summary only.")
        args.summary = True

    # Load metrics
    print(f"Loading metrics from: {args.experiment_dir}")
    loader = TrainingMetricsLoader(args.experiment_dir)
    loader.load_all()

    exp_names = list(loader.metrics.keys())
    if not exp_names:
        print("ERROR: No experiments found in specified directories.")
        print("Check the directory structure:")
        print("  experiment_dir/")
        print("    ├── config.json")
        print("    ├── models/")
        print("    ├── checkpoints/")
        print("    ├── metrics/")
        print("    │   ├── eval_metrics.csv")
        print("    │   └── final_eval.json")
        print("    └── logs/")
        print("        └── tensorboard/")
        return

    # Visualizer
    fig_size = tuple(args.figure_size)
    visualizer = MetricsVisualizer(
        font_size=args.font_size,
        figure_size=fig_size,
        output_dir=args.output_dir if not args.no_save else None,
    )

    # Print summary
    if args.summary:
        visualizer.print_summary(loader)

    # Create plots
    if args.metrics[0] == "all" or HAS_MATPLOTLIB:
        figures = visualizer.plot_comparison(loader, exp_names)

        # Save figures
        if not args.no_save and args.output_dir:
            visualizer.save_figures(figures, exp_names)

        # Show figures
        if args.show:
            visualizer.show_figures(figures)
        elif not args.output_dir:
            visualizer.show_figures(figures)
    else:
        # Plot specific metrics
        for metric in args.metrics:
            if metric == "loss":
                fig = visualizer.plot_loss_curves(loader, exp_names)
                if not args.no_save and args.output_dir:
                    visualizer.save_figures([fig], exp_names)
                if args.show:
                    visualizer.show_figures([fig])

            elif metric == "epsilon":
                fig = visualizer.plot_epsilon_decay(loader, exp_names)
                if not args.no_save and args.output_dir:
                    visualizer.save_figures([fig], exp_names)
                if args.show:
                    visualizer.show_figures([fig])

            elif metric == "reward":
                fig = visualizer.plot_eval_rewards(loader, exp_names)
                if not args.no_save and args.output_dir:
                    visualizer.save_figures([fig], exp_names)
                if args.show:
                    visualizer.show_figures([fig])

            elif metric == "buffer":
                fig = visualizer.plot_buffer_size(loader, exp_names)
                if not args.no_save and args.output_dir:
                    visualizer.save_figures([fig], exp_names)
                if args.show:
                    visualizer.show_figures([fig])

            elif metric == "self_play":
                fig = visualizer.plot_self_play_episode_metrics(loader, exp_names)
                if fig is not None:
                    if not args.no_save and args.output_dir:
                        visualizer.save_figures([fig], exp_names)
                    if args.show:
                        visualizer.show_figures([fig])

            elif metric == "win_rate":
                fig = visualizer.plot_win_rate_eval(loader, exp_names)
                if fig is not None:
                    if not args.no_save and args.output_dir:
                        visualizer.save_figures([fig], exp_names)
                    if args.show:
                        visualizer.show_figures([fig])

    print("\nVisualization complete!")
    if args.output_dir and not args.no_save:
        print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
