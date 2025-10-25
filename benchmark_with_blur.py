#!/usr/bin/env python3
"""Quick benchmark to test --background no,blur=25 configuration"""

import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lfbw'))

from lfbw import ImageSegmenter, getPercentageFloat, create_filter_config, process_filter_args, getNextOddNumber
import cv2
import numpy as np

def create_synthetic_frame(width, height):
    """Create a synthetic BGR frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        frame[y, :, 0] = int(255 * y / height)
        frame[y, :, 1] = int(128 * y / height)
    for x in range(width):
        frame[:, x, 2] = int(255 * x / width)
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(frame, (center_x, center_y), (width // 8, height // 4),
                0, 0, 360, (180, 150, 120), -1)
    return frame

def benchmark_config(width, height, background_config, num_frames=50, warmup=5):
    """Benchmark with specific background configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {width}x{height} with --background {background_config}")
    print(f"{'='*60}\n")

    # Initialize
    segmenter = ImageSegmenter(width, height)
    filters = {
        'background': create_filter_config(process_filter_args(background_config), 'background')
    }

    bg_config = filters['background']
    threshold = 0.75
    postprocess = True

    # Create frame
    frame = create_synthetic_frame(width, height)

    # Warmup
    print(f"Warming up ({warmup} frames)...")
    for _ in range(warmup):
        mask = segmenter.segment(frame)
        if threshold < 1:
            cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY, dst=mask)
        if postprocess:
            cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1, dst=mask)
            cv2.GaussianBlur(mask, (7, 7), 0, dst=mask)

        # Background generation
        if bg_config['disabled']:
            background_frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            blur_val = bg_config['blur'] if bg_config['blur'] is not None else 15
            blur_val = getNextOddNumber(blur_val)
            sigma = blur_val / 3
            background_frame = cv2.GaussianBlur(frame, (blur_val, blur_val), sigma)

    # Benchmark
    print(f"Benchmarking ({num_frames} frames)...")
    times = []
    times_breakdown = {
        'segment': [],
        'postprocess': [],
        'background': [],
        'total': []
    }

    for i in range(num_frames):
        t_start = time.perf_counter()

        # Segmentation
        t0 = time.perf_counter()
        mask = segmenter.segment(frame)
        t1 = time.perf_counter()
        times_breakdown['segment'].append((t1 - t0) * 1000)

        # Threshold + postprocess
        if threshold < 1:
            cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY, dst=mask)
        if postprocess:
            cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1, dst=mask)
            cv2.GaussianBlur(mask, (7, 7), 0, dst=mask)
        t2 = time.perf_counter()
        times_breakdown['postprocess'].append((t2 - t1) * 1000)

        # Background generation
        if bg_config['disabled']:
            background_frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            blur_val = bg_config['blur'] if bg_config['blur'] is not None else 15
            blur_val = getNextOddNumber(blur_val)
            sigma = blur_val / 3
            background_frame = cv2.GaussianBlur(frame, (blur_val, blur_val), sigma)
        t3 = time.perf_counter()
        times_breakdown['background'].append((t3 - t2) * 1000)

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000
        times.append(elapsed_ms)
        times_breakdown['total'].append(elapsed_ms)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_frames} frames", end='\r')

    print(f"  Progress: {num_frames}/{num_frames} frames")

    # Statistics
    avg_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    avg_fps = 1000 / avg_time
    max_fps = 1000 / min_time
    min_fps = 1000 / max_time

    avg_segment = statistics.mean(times_breakdown['segment'])
    avg_postprocess = statistics.mean(times_breakdown['postprocess'])
    avg_background = statistics.mean(times_breakdown['background'])

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total time:    {avg_time:6.2f}ms (stdev: {stdev_time:5.2f}ms)")
    print(f"  Segment:     {avg_segment:6.2f}ms ({avg_segment/avg_time*100:5.1f}%)")
    print(f"  Postprocess: {avg_postprocess:6.2f}ms ({avg_postprocess/avg_time*100:5.1f}%)")
    print(f"  Background:  {avg_background:6.2f}ms ({avg_background/avg_time*100:5.1f}%)")
    print(f"\nMin time:      {min_time:6.2f}ms (max fps: {max_fps:5.1f})")
    print(f"Max time:      {max_time:6.2f}ms (min fps: {min_fps:5.1f})")
    print(f"\nAverage FPS:   {avg_fps:6.2f}")
    print(f"Target FPS:    30.00")
    print(f"Performance:   {(avg_fps/30)*100:.1f}% of target")

    if avg_fps >= 30:
        print(f"\n✓ PASS: Achieving target 30fps")
    else:
        shortfall = 30 - avg_fps
        needed_speedup = 30 / avg_fps
        print(f"\n✗ FAIL: {shortfall:.1f} fps below target")
        print(f"Need {(needed_speedup - 1) * 100:.1f}% speedup to reach 30fps")

    print(f"{'='*60}\n")

    segmenter.close()

    return {
        'avg_fps': avg_fps,
        'avg_time': avg_time,
        'breakdown': {
            'segment': avg_segment,
            'postprocess': avg_postprocess,
            'background': avg_background
        }
    }

if __name__ == '__main__':
    import platform

    print("\nLFBW Background Blur Comparison")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}\n")

    width, height = 1920, 1080

    # Test 1: --background no (optimized)
    result1 = benchmark_config(width, height, "no", num_frames=50, warmup=5)

    # Test 2: --background no,blur=25 (with blur)
    result2 = benchmark_config(width, height, "no,blur=25", num_frames=50, warmup=5)

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Configuration':<30} {'FPS':>10} {'Frame Time':>15}")
    print(f"{'-'*60}")
    print(f"{'--background no':<30} {result1['avg_fps']:>10.2f} {result1['avg_time']:>12.2f}ms")
    print(f"{'--background no,blur=25':<30} {result2['avg_fps']:>10.2f} {result2['avg_time']:>12.2f}ms")
    print(f"{'-'*60}")

    fps_diff = result1['avg_fps'] - result2['avg_fps']
    time_diff = result2['avg_time'] - result1['avg_time']
    bg_time_diff = result2['breakdown']['background'] - result1['breakdown']['background']

    print(f"\nDifference: {fps_diff:+.2f} fps ({time_diff:+.2f}ms per frame)")
    print(f"Background blur overhead: {bg_time_diff:+.2f}ms")
    print(f"{'='*60}\n")
