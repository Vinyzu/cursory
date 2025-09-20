# IMPORTANT: Playwright-Python runs via a Node.js backend, so Delay times are commonly higher than the timings.
# Therefore, you probably should not use Playwright for super high precision mouse movements.
# You probably should use something like [CDP-Patches](https://github.com/Kaliiiiiiiiii-Vinyzu/CDP-Patches/).
import random
import time

from cursory import generate_trajectory
from playwright.sync_api import Playwright, sync_playwright


def run(playwright: Playwright) -> None:
    """Run a sample test using Playwright."""
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()

    page.goto("https://google.com/blank.html")
    # Inject mouse tracking script
    with open("show_mouse.html") as f:
        page.set_content(f.read())

    end_point = (0, 0)
    for _ in range(10):
        # Generate a random trajectory
        start_point = end_point
        end_point = (random.randint(0, 1080), random.randint(0, 720))
        trajectory_points, timings = generate_trajectory(start_point, end_point, frequency_randomizer=0, frequency=100)
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
            page.wait_for_timeout(pause_time - delay_time)

        # Click at the end point
        page.mouse.click(end_point[0], end_point[1])
        page.wait_for_timeout(1000)

    browser.close()


with sync_playwright() as playwright:
    run(playwright)
