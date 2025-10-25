#!/usr/bin/env python3
"""
Benchmark script for LFBW processing layer.

This script benchmarks the frame processing performance without requiring
webcam devices or v4l2loopback setup. It tests with synthetic frames.

Usage:
    python3 benchmark_processing.py
    python3 benchmark_processing.py --width 1920 --height 1080
    python3 benchmark_processing.py --profile
"""

import sys
import time
import argparse
import numpy as np
import statistics
from pathlib import Path

# Add lfbw to path
sys.path.insert(0, str(Path(__file__).parent / 'lfbw'))

from lfbw import ImageSegmenter


def create_synthetic_frame(width, height):
    """Create a synthetic BGR frame with some content."""
    # Create a frame with gradient and noise for realistic processing
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add gradient
    for y in range(height):
        frame[y, :, 0] = int(255 * y / height)  # B
        frame[y, :, 1] = int(128 * y / height)  # G

    for x in range(width):
        frame[:, x, 2] = int(255 * x / width)  # R

    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Add a person-like shape in the center
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(frame, (center_x, center_y), (width // 8, height // 4),
                0, 0, 360, (180, 150, 120), -1)

    return frame


def benchmark_segmentation(width, height, num_frames=100, warmup=10):
    """Benchmark the segmentation process."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {width}x{height} segmentation")
    print(f"{'='*60}\n")

    # Create segmenter
    print("Initializing ImageSegmenter...")
    segmenter = ImageSegmenter(width, height)

    # Create synthetic frame
    print(f"Creating synthetic {width}x{height} frame...")
    frame = create_synthetic_frame(width, height)

    # Warmup
    print(f"Warming up ({warmup} frames)...")
    for _ in range(warmup):
        _ = segmenter.segment(frame)

    # Benchmark
    print(f"Benchmarking ({num_frames} frames)...")
    times = []

    for i in range(num_frames):
        t_start = time.perf_counter()
        mask = segmenter.segment(frame)
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000
        times.append(elapsed_ms)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_frames} frames", end='\r')

    print(f"  Progress: {num_frames}/{num_frames} frames")

    # Statistics
    avg_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    p50_time = statistics.median(times)

    avg_fps = 1000 / avg_time
    max_fps = 1000 / min_time
    min_fps = 1000 / max_time

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Average time:  {avg_time:6.2f}ms (stdev: {stdev_time:5.2f}ms)")
    print(f"Min time:      {min_time:6.2f}ms (max fps: {max_fps:5.1f})")
    print(f"Max time:      {max_time:6.2f}ms (min fps: {min_fps:5.1f})")
    print(f"Median time:   {p50_time:6.2f}ms")
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

    # Cleanup
    segmenter.close()

    return {
        'avg_time': avg_time,
        'avg_fps': avg_fps,
        'stdev_time': stdev_time,
        'times': times
    }


def benchmark_full_pipeline(width, height, num_frames=100, warmup=10):
    """Benchmark the full processing pipeline (segmentation + compositing)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {width}x{height} full pipeline")
    print(f"Config: --background no --mask no")
    print(f"{'='*60}\n")

    # Create a mock args object for FakeCam
    class MockArgs:
        def __init__(self):
            self.webcam_path = "/dev/video0"
            self.width = width
            self.height = height
            self.fps = 30
            self.codec = "MJPG"
            self.use_sigmoid = False
            self.threshold = 75
            self.no_postprocess = True  # Enable postprocessing
            self.no_ondemand = True
            self.v4l2loopback_path = "/dev/video2"
            self.selfie = ""
            self.background = "no"  # User's requirement
            self.mask = "no"  # User's requirement
            self.dump = False
            self.profile = False
            self.select_model = 1

    args = MockArgs()

    print("Initializing processing pipeline...")
    from lfbw import ImageSegmenter, getPercentageFloat, create_filter_config, process_filter_args
    import cv2

    # Initialize segmenter
    segmenter = ImageSegmenter(width, height)

    # Process filter configs
    filters = {
        'selfie': create_filter_config(process_filter_args(args.selfie), 'selfie'),
        'background': create_filter_config(process_filter_args(args.background), 'background'),
        'mask': create_filter_config(process_filter_args(args.mask), 'mask')
    }

    threshold = getPercentageFloat(args.threshold)
    use_sigmoid = args.use_sigmoid
    postprocess = args.no_postprocess

    # Create synthetic frame
    print(f"Creating synthetic {width}x{height} frame...")
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

    # Benchmark
    print(f"Benchmarking ({num_frames} frames)...")
    times = []
    times_breakdown = {
        'segment': [],
        'threshold': [],
        'postprocess': []
    }

    for i in range(num_frames):
        t_start = time.perf_counter()

        # Segmentation
        t0 = time.perf_counter()
        mask = segmenter.segment(frame)
        t1 = time.perf_counter()
        times_breakdown['segment'].append((t1 - t0) * 1000)

        # Threshold
        if threshold < 1:
            cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY, dst=mask)
        t2 = time.perf_counter()
        times_breakdown['threshold'].append((t2 - t1) * 1000)

        # Postprocess
        if postprocess:
            cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1, dst=mask)
            cv2.GaussianBlur(mask, (7, 7), 0, dst=mask)
        t3 = time.perf_counter()
        times_breakdown['postprocess'].append((t3 - t2) * 1000)

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000
        times.append(elapsed_ms)

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

    # Breakdown statistics
    avg_segment = statistics.mean(times_breakdown['segment'])
    avg_threshold = statistics.mean(times_breakdown['threshold'])
    avg_postprocess = statistics.mean(times_breakdown['postprocess'])

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total time:    {avg_time:6.2f}ms (stdev: {stdev_time:5.2f}ms)")
    print(f"  Segment:     {avg_segment:6.2f}ms ({avg_segment/avg_time*100:5.1f}%)")
    print(f"  Threshold:   {avg_threshold:6.2f}ms ({avg_threshold/avg_time*100:5.1f}%)")
    print(f"  Postprocess: {avg_postprocess:6.2f}ms ({avg_postprocess/avg_time*100:5.1f}%)")
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
        print(f"\nBottleneck analysis:")
        if avg_segment > avg_time * 0.7:
            print(f"  → Segmentation is the bottleneck ({avg_segment/avg_time*100:.1f}% of time)")
        if avg_postprocess > avg_time * 0.2:
            print(f"  → Postprocessing is significant ({avg_postprocess/avg_time*100:.1f}% of time)")

    print(f"{'='*60}\n")

    # Cleanup
    segmenter.close()

    return {
        'avg_time': avg_time,
        'avg_fps': avg_fps,
        'stdev_time': stdev_time,
        'breakdown': {
            'segment': avg_segment,
            'threshold': avg_threshold,
            'postprocess': avg_postprocess
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark LFBW processing performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 benchmark_processing.py
  python3 benchmark_processing.py --width 1920 --height 1080
  python3 benchmark_processing.py -n 200 --warmup 20
        '''
    )
    parser.add_argument('-W', '--width', type=int, default=1920,
                        help='Frame width (default: 1920)')
    parser.add_argument('-H', '--height', type=int, default=1080,
                        help='Frame height (default: 1080)')
    parser.add_argument('-n', '--num-frames', type=int, default=100,
                        help='Number of frames to benchmark (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup frames (default: 10)')
    parser.add_argument('--segmentation-only', action='store_true',
                        help='Benchmark only segmentation (skip pipeline)')
    parser.add_argument('--pipeline-only', action='store_true',
                        help='Benchmark only full pipeline (skip segmentation)')

    args = parser.parse_args()

    # Import cv2 here so we can show error if not available
    global cv2
    import cv2

    print(f"\nLFBW Processing Benchmark")
    print(f"Target config: -W {args.width} -H {args.height} -F 30 --background no --mask no")
    print(f"Frames: {args.num_frames}, Warmup: {args.warmup}\n")

    # System info
    import platform
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")

    try:
        import mediapipe as mp
        print(f"MediaPipe: {mp.__version__}")
    except:
        print("MediaPipe: not available")

    results = {}

    # Run segmentation benchmark
    if not args.pipeline_only:
        results['segmentation'] = benchmark_segmentation(
            args.width, args.height, args.num_frames, args.warmup
        )

    # Run full pipeline benchmark
    if not args.segmentation_only:
        results['pipeline'] = benchmark_full_pipeline(
            args.width, args.height, args.num_frames, args.warmup
        )

    # Summary
    if not args.segmentation_only and not args.pipeline_only:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        seg_fps = results['segmentation']['avg_fps']
        pipe_fps = results['pipeline']['avg_fps']
        print(f"Segmentation only:  {seg_fps:6.2f} fps")
        print(f"Full pipeline:      {pipe_fps:6.2f} fps")
        overhead = seg_fps - pipe_fps
        print(f"Pipeline overhead:  {overhead:6.2f} fps ({overhead/seg_fps*100:.1f}%)")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
