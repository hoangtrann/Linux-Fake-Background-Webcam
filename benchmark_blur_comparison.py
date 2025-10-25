#!/usr/bin/env python3
"""Benchmark: --background no vs blur background (blur=25)"""

import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lfbw'))

from lfbw import ImageSegmenter, create_filter_config, process_filter_args, getNextOddNumber
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

def benchmark_background(width, height, bg_config_str, num_frames=50, warmup=5):
    """Benchmark with specific background configuration."""
    print(f"\n{'='*60}")
    print(f"Config: --background {bg_config_str}")
    print(f"{'='*60}\n")

    segmenter = ImageSegmenter(width, height)
    bg_config = create_filter_config(process_filter_args(bg_config_str), 'background')
    frame = create_synthetic_frame(width, height)

    # Warmup
    print(f"Warmup ({warmup} frames)...")
    for _ in range(warmup):
        mask = segmenter.segment(frame)
        cv2.threshold(mask, 0.75, 1, cv2.THRESH_BINARY, dst=mask)
        cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1, dst=mask)
        cv2.GaussianBlur(mask, (7, 7), 0, dst=mask)

        # Generate background based on config
        if bg_config['disabled']:
            # Solid black - optimized path
            background = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Blur background - unoptimized path
            blur_val = bg_config['blur'] if bg_config['blur'] is not None else 15
            blur_val = getNextOddNumber(blur_val)
            sigma = blur_val / 3
            background = cv2.GaussianBlur(frame, (blur_val, blur_val), sigma)

    # Benchmark
    print(f"Benchmark ({num_frames} frames)...")
    times_seg = []
    times_post = []
    times_bg = []
    times_total = []

    for i in range(num_frames):
        t0 = time.perf_counter()

        # Segmentation
        mask = segmenter.segment(frame)
        t1 = time.perf_counter()

        # Post-processing
        cv2.threshold(mask, 0.75, 1, cv2.THRESH_BINARY, dst=mask)
        cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1, dst=mask)
        cv2.GaussianBlur(mask, (7, 7), 0, dst=mask)
        t2 = time.perf_counter()

        # Background generation
        if bg_config['disabled']:
            background = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            blur_val = bg_config['blur'] if bg_config['blur'] is not None else 15
            blur_val = getNextOddNumber(blur_val)
            sigma = blur_val / 3
            background = cv2.GaussianBlur(frame, (blur_val, blur_val), sigma)
        t3 = time.perf_counter()

        times_seg.append((t1 - t0) * 1000)
        times_post.append((t2 - t1) * 1000)
        times_bg.append((t3 - t2) * 1000)
        times_total.append((t3 - t0) * 1000)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_frames}", end='\r')

    print(f"  {num_frames}/{num_frames}")

    avg_seg = statistics.mean(times_seg)
    avg_post = statistics.mean(times_post)
    avg_bg = statistics.mean(times_bg)
    avg_total = statistics.mean(times_total)
    stdev_total = statistics.stdev(times_total)
    avg_fps = 1000 / avg_total

    print(f"\nResults:")
    print(f"  Segmentation:    {avg_seg:6.2f}ms")
    print(f"  Post-processing: {avg_post:6.2f}ms")
    print(f"  Background blur: {avg_bg:6.2f}ms")
    print(f"  Total:           {avg_total:6.2f}ms (±{stdev_total:.2f}ms)")
    print(f"  Average FPS:     {avg_fps:6.2f}")

    status = "✓ PASS" if avg_fps >= 30 else f"✗ FAIL ({30-avg_fps:.1f} fps short)"
    print(f"  Target 30fps:    {status}")

    segmenter.close()
    return {
        'fps': avg_fps,
        'total': avg_total,
        'seg': avg_seg,
        'post': avg_post,
        'bg': avg_bg,
        'stdev': stdev_total
    }

if __name__ == '__main__':
    import platform

    print("\n" + "="*60)
    print("LFBW Background Configuration Comparison")
    print("="*60)
    print(f"Resolution: 1920x1080 @ 30fps target")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}, OpenCV: {cv2.__version__}")

    width, height = 1920, 1080

    # Test 1: Optimized - no background (black)
    r1 = benchmark_background(width, height, "no", num_frames=50, warmup=5)

    # Test 2: Blur background with blur=25
    r2 = benchmark_background(width, height, "blur=25", num_frames=50, warmup=5)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<30} {'FPS':>10} {'Frame Time':>12} {'BG Time':>10}")
    print(f"{'-'*60}")
    print(f"{'--background no':<30} {r1['fps']:>10.2f} {r1['total']:>10.2f}ms {r1['bg']:>9.2f}ms")
    print(f"{'--background blur=25':<30} {r2['fps']:>10.2f} {r2['total']:>10.2f}ms {r2['bg']:>9.2f}ms")
    print(f"{'-'*60}")

    fps_loss = r1['fps'] - r2['fps']
    time_cost = r2['total'] - r1['total']
    bg_cost = r2['bg'] - r1['bg']

    print(f"\nBlur overhead:")
    print(f"  Background time: +{bg_cost:.2f}ms")
    print(f"  Total frame time: +{time_cost:.2f}ms")
    print(f"  FPS impact: {fps_loss:+.2f} fps")
    print(f"\nConclusion:")
    if r2['fps'] >= 30:
        print(f"  ✓ blur=25 achieves 30fps target ({r2['fps']:.1f} fps)")
    else:
        print(f"  ✗ blur=25 falls short of 30fps ({r2['fps']:.1f} fps)")
    print(f"{'='*60}\n")
