import matplotlib.pyplot as plt
import numpy as np
from cursory import Point


def plot_trajectories_grid(
    all_trajectories: list[tuple[list[Point], list[int], Point, Point]],
) -> None:
    """Plot original and morphed trajectories in a grid for visual comparison.

    Args:
        all_trajectories (List[Tuple[List[Point], List[int], Point, Point]]):
            List of tuples, where each tuple contains:
            - Trajectory points (List[Point]).
            - Timings (List[int]).
            - Target start point (Point).
            - Target end point (Point).
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, (trajectory_points, _, target_start, target_end) in enumerate(all_trajectories):
        ax = axes[i]

        # Convert point lists to numpy arrays for efficient plotting
        trajectory_points_np = np.array(trajectory_points)

        # Plot the generated trajectory in a blue line
        ax.plot(
            trajectory_points_np[:, 0],
            trajectory_points_np[:, 1],
            "b-",
            linewidth=2,
            label="Morphed",
        )
        ax.scatter(
            trajectory_points_np[:, 0],
            trajectory_points_np[:, 1],
            color="blue",
            s=10,
            label="Morphed Points",
        )

        # Plot target start and end points as larger green and red scatters
        ax.scatter(
            [target_start[0]],
            [target_start[1]],
            color="green",
            s=50,
            label="Start",
            zorder=3,
        )
        ax.scatter([target_end[0]], [target_end[1]], color="red", s=50, label="End", zorder=3)

        ax.set_title(f"Trajectory {i + 1}")
        ax.legend(fontsize="x-small")
        ax.grid(True)  # noqa: FBT003
        ax.axis("equal")

    plt.tight_layout()
    plt.show()


def plot_timings_grid(
    all_trajectories: list[tuple[list[Point], list[int], Point, Point]],
) -> None:
    """Plot timings for each trajectory in a grid.

    Args:
        all_trajectories (List[Tuple[List[Point], List[int], Point, Point]]):
            List of tuples, where each tuple contains:
            - Trajectory points (List[Point]).
            - Timings (List[int]).
            - Target start point (Point).
            - Target end point (Point).
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, (_, timings, _, _) in enumerate(all_trajectories):
        ax = axes[i]
        timings_converted = [timings[i] - timings[i + 1] for i in range(len(timings) - 1)]
        # Convert timings to numpy array for efficient plotting
        timings_np = np.array(timings_converted)

        # Plot timings as a line graph
        ax.plot(timings_np, marker="o", linestyle="-", color="blue", label="Timings")

        # Highlight start and end points
        ax.axhline(y=timings_np[0], color="green", linestyle="--", label="Start Timing")
        ax.axhline(y=timings_np[-1], color="red", linestyle="--", label="End Timing")

        ax.set_title(f"Timings {i + 1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Timing (ms)")
        ax.legend(fontsize="x-small")
        ax.grid(True)  # noqa: FBT003

    plt.tight_layout()
    plt.show()


def plot_trajectories_overlayed(
    all_trajectories: list[tuple[list[Point], list[int], Point, Point]],
) -> None:
    """Plot all morphed trajectories overlayed in a single plot for comparison.

    Args:
        all_trajectories (List[Tuple[List[Point], List[int], Point, Point]]):
            List of tuples, where each tuple contains:
            - Trajectory points (List[Point]).
            - Timings (List[int]).
            - Target start point (Point).
            - Target end point (Point).
    """
    plt.figure(figsize=(10, 8))

    for i, (trajectory_points, _, target_start, target_end) in enumerate(all_trajectories):
        # Convert point lists to numpy arrays for efficient plotting
        trajectory_points_np = np.array(trajectory_points)

        # Plot the generated trajectory in a blue line
        plt.plot(
            trajectory_points_np[:, 0],
            trajectory_points_np[:, 1],
            linewidth=1,
            alpha=0.7,
            label=f"Trajectory {i + 1}",
        )
        plt.scatter(
            trajectory_points_np[:, 0],
            trajectory_points_np[:, 1],
            s=5,
            alpha=0.7,
        )

        # Plot target start and end points as larger green and red scatters
        plt.scatter(
            [target_start[0]],
            [target_start[1]],
            color="green",
            s=100,
            label=f"Start {i + 1}",
            zorder=3,
        )
        plt.scatter([target_end[0]], [target_end[1]], color="red", s=100, label=f"End {i + 1}", zorder=3)

    plt.title("Overlayed Morphed Trajectories")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)  # noqa: FBT003
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
