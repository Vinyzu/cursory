import random
import time

from cursory import Point, generate_trajectory
from plot_trajectories import plot_timings_grid, plot_trajectories_grid, plot_trajectories_overlayed

if __name__ == "__main__":
    generated_trajectories: list[tuple[list[Point], list[int], Point, Point]] = []
    num_trajectories_to_generate = 10
    times = 0.0

    for _ in range(num_trajectories_to_generate):
        # Generate two random points on a 1920x1080 screen
        target_start = (random.randint(0, 1920), random.randint(0, 1080))
        target_end = (random.randint(0, 1920), random.randint(0, 1080))

        # target_start = (200, 100)
        # target_end = (800, 900) #  +random.randint(0, 10)

        start_time = time.time()  # Record start time
        generated_trajectory, generated_timings = generate_trajectory(target_start, target_end)
        times += time.time() - start_time

        generated_trajectories.append((generated_trajectory, generated_timings, target_start, target_end))

        mact = ""  # Format: idx,action,timestamp,x,y;
        for idx, (point, timing) in enumerate(zip(generated_trajectory, generated_timings, strict=False)):
            mact += f"{idx},move,{timing},{int(point[0])},{int(point[1])};"
        # print(mact)

    print("Average time:", times / num_trajectories_to_generate)
    plot_trajectories_grid(generated_trajectories)
    plot_trajectories_overlayed(generated_trajectories)
    plot_timings_grid(generated_trajectories)
