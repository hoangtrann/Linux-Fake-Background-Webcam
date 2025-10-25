#!/bin/bash
# Quick test script to verify optimizations are working
# This compares --background no vs default blur background

echo "Testing CPU optimizations for --background no"
echo "=============================================="
echo ""

# Check if dependencies are available
if ! python3 -c "import cv2, mediapipe, numpy" 2>/dev/null; then
    echo "Error: Missing dependencies (opencv-python, mediapipe, numpy)"
    echo "Please install: pip install -r requirements.txt"
    exit 1
fi

echo "Test 1: With --background no (optimized path)"
echo "----------------------------------------------"
timeout 30 python3 -c "
import sys
sys.path.insert(0, 'lfbw')
from lfbw import ImageSegmenter, create_filter_config, process_filter_args, getPercentageFloat
import cv2
import numpy as np
import time

# Create segmenter
width, height = 1920, 1080
segmenter = ImageSegmenter(width, height)

# Create synthetic frame
frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Parse config
filters = {
    'background': create_filter_config(process_filter_args('no'), 'background')
}

# Warmup
for _ in range(10):
    mask = segmenter.segment(frame)

# Benchmark
times = []
for i in range(50):
    t0 = time.perf_counter()

    mask = segmenter.segment(frame)

    # Simulate the background generation (optimized path)
    bg_config = filters['background']
    if bg_config['disabled']:
        background_frame = np.zeros((height, width, 3), dtype=np.uint8)

    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

    if (i + 1) % 10 == 0:
        print(f'  Progress: {i+1}/50', end='\r')

print(f'  Progress: 50/50')
avg_time = sum(times) / len(times)
avg_fps = 1000 / avg_time
print(f'\nAverage time: {avg_time:.2f}ms')
print(f'Average FPS:  {avg_fps:.2f}')
print(f'Target: 30fps - ', end='')
if avg_fps >= 30:
    print('✓ PASS')
else:
    print(f'✗ FAIL (need {30 - avg_fps:.1f} more fps)')

segmenter.close()
"

echo ""
echo "Test 2: With default blur background (unoptimized)"
echo "---------------------------------------------------"
timeout 30 python3 -c "
import sys
sys.path.insert(0, 'lfbw')
from lfbw import ImageSegmenter
import cv2
import numpy as np
import time

# Create segmenter
width, height = 1920, 1080
segmenter = ImageSegmenter(width, height)

# Create synthetic frame
frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Warmup
for _ in range(10):
    mask = segmenter.segment(frame)
    bg = cv2.GaussianBlur(frame, (15, 15), 5)

# Benchmark
times = []
for i in range(50):
    t0 = time.perf_counter()

    mask = segmenter.segment(frame)

    # Simulate the background generation (unoptimized path with blur)
    background_frame = cv2.GaussianBlur(frame, (15, 15), 5)

    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

    if (i + 1) % 10 == 0:
        print(f'  Progress: {i+1}/50', end='\r')

print(f'  Progress: 50/50')
avg_time = sum(times) / len(times)
avg_fps = 1000 / avg_time
print(f'\nAverage time: {avg_time:.2f}ms')
print(f'Average FPS:  {avg_fps:.2f}')
print(f'Target: 30fps - ', end='')
if avg_fps >= 30:
    print('✓ PASS')
else:
    print(f'✗ FAIL (need {30 - avg_fps:.1f} more fps)')

segmenter.close()
"

echo ""
echo "=============================================="
echo "Summary: Test 1 should be significantly faster"
echo "Expected improvement: ~15ms per frame"
