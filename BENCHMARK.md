# Benchmarking Guide

This repository includes two benchmark tools:

## 1. `benchmark.sh` - Full System Benchmark

Tests the complete pipeline including webcam input and v4l2loopback output.

**Requirements:**
- v4l2loopback kernel module
- ffmpeg or avconv
- Internet connection (downloads test video)

**Usage:**
```bash
./benchmark.sh [additional lfbw.py arguments]
```

**What it does:**
- Creates v4l2loopback devices
- Downloads and loops a test video to /dev/video99
- Runs lfbw.py for 180 seconds
- Calculates average FPS and standard deviation

## 2. `benchmark_processing.py` - Processing Layer Benchmark

Tests only the processing performance without hardware dependencies. **Recommended for optimization work.**

**Requirements:**
- Python 3
- opencv-python
- mediapipe
- numpy

**Usage:**
```bash
# Default: 1920x1080, 100 frames
python3 benchmark_processing.py

# Custom resolution
python3 benchmark_processing.py -W 1280 -H 720

# More frames for better accuracy
python3 benchmark_processing.py -n 200 --warmup 20

# Benchmark only segmentation
python3 benchmark_processing.py --segmentation-only

# Benchmark full pipeline (segmentation + post-processing)
python3 benchmark_processing.py --pipeline-only
```

**What it measures:**
- **Segmentation only**: MediaPipe inference + mask upscaling + blur
- **Full pipeline**: Segmentation + threshold + dilate + blur (simulates `--background no --mask no`)

**Output includes:**
- Average/min/max frame times
- Standard deviation
- FPS (average, min, max)
- Performance vs 30fps target
- Detailed breakdown of time spent in each operation
- Bottleneck identification

**Example output:**
```
============================================================
Benchmarking 1920x1080 full pipeline
Config: --background no --mask no
============================================================

...

============================================================
RESULTS
============================================================
Total time:    33.45ms (stdev:  2.31ms)
  Segment:     28.12ms ( 84.1%)
  Threshold:    0.23ms (  0.7%)
  Postprocess:  5.10ms ( 15.2%)

Average FPS:   29.89
Target FPS:    30.00
Performance:   99.6% of target

✓ PASS: Achieving target 30fps
============================================================
```

## Benchmarking Workflow

1. **Baseline**: Run benchmark with current code
   ```bash
   python3 benchmark_processing.py > baseline.txt
   ```

2. **Make optimizations**: Modify code

3. **Compare**: Run benchmark again
   ```bash
   python3 benchmark_processing.py > optimized.txt
   diff baseline.txt optimized.txt
   ```

4. **Iterate**: Repeat until target FPS is achieved

## Target Configurations

### High Resolution (1920x1080 @ 30fps)
```bash
python3 benchmark_processing.py -W 1920 -H 1080
```
Target: 30+ fps

### Standard Resolution (1280x720 @ 30fps)
```bash
python3 benchmark_processing.py -W 1280 -H 720
```
Target: 60+ fps

### Low Resolution (640x480 @ 30fps)
```bash
python3 benchmark_processing.py -W 640 -H 480
```
Target: 120+ fps

## Interpreting Results

### If FPS is below target:
- Check the breakdown to identify bottlenecks
- Segmentation >70%: MediaPipe inference is slow
  - Consider smaller input resolution to MediaPipe
  - GPU acceleration may help
- Postprocess >20%: Dilate/blur operations are slow
  - Reduce kernel sizes
  - Skip postprocessing (`--no-postprocess` flag)
- Check CPU usage (thermal throttling?)

### Optimization priorities:
1. Fix operations that take >10ms
2. Focus on operations that take >50% of total time
3. Consider trade-offs between quality and performance
