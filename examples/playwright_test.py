# IMPORTANT: Playwright-Python runs via a Node.js backend, so Delay times are commonly higher than the timings.
# Therefore, you probably should not use Playwright for super high precision mouse movements.
# You probably should use something like [CDP-Patches](https://github.com/Kaliiiiiiiiii-Vinyzu/CDP-Patches/).
import random
import time

from cursory import generate_trajectory
from playwright.sync_api import Playwright, sync_playwright
from selenium_driverless.utils.utils import sel_driverless_path  # type: ignore[import-untyped]


def high_precision_sleep(duration: float) -> None:
    """Wait for a duration with higher precision than a single sleep call."""
    start_time = time.perf_counter()
    while True:
        elapsed_time = time.perf_counter() - start_time
        remaining_time = duration - elapsed_time
        if remaining_time <= 0:
            break
        if remaining_time > 0.02:  # Sleep for 5ms if remaining time is greater
            time.sleep(max(remaining_time / 2, 0.0001))  # Sleep for the remaining time or minimum sleep interval
        else:
            pass


def run(playwright: Playwright) -> None:
    """Run a sample test using Playwright."""
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto("https://google.com/blank.html")
    # Inject mouse tracking script
    page.add_script_tag(path=sel_driverless_path() + "files/js/show_mousemove.js")

    end_point = (0, 0)
    total_points = 0
    timeout_points = 0

    for _ in range(10):
        # Generate a random trajectory
        start_point = end_point
        end_point = (random.randint(0, 1080), random.randint(0, 720))
        trajectory_points, timings = generate_trajectory(start_point, end_point, frequency_randomizer=0, frequency=60)
        # Print Frequency
        avg_freq = 1000 / (timings[-1] - timings[0]) * len(timings)
        print(f"Average Frequency: {avg_freq:.2f} Hz")

        # Simulate drawing the trajectory on the page
        for i, (point, timing) in enumerate(zip(trajectory_points, timings, strict=False)):
            # Move mouse to the point and measure delay
            start = time.time()
            page.mouse.move(point[0], point[1])
            delay_time = (time.time() - start) * 1000

            # Wait for the remaining time in the timing interval
            pause_time = timing - (timings[i - 1] if i > 0 else 0)
            if delay_time > pause_time:
                print(f"[WARNING]: Delay ({delay_time:.2f} ms) is greater than timing pause ({pause_time} ms)")
                timeout_points += 1
            total_points += 1

            sleep_delay = max(((pause_time - delay_time) / 1000), 0)
            # page.wait_for_timeout(pause_time - delay_time)
            high_precision_sleep(sleep_delay)

        # Click at the end point
        page.mouse.click(end_point[0], end_point[1])
        page.wait_for_timeout(1000)

    percentage = (timeout_points / total_points) * 100
    print(f"Total Points: {total_points}, Timeout Points: {timeout_points}, Percentage: {percentage:.2f}%")
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
