# Camera Calibration with Checkerboard

A Python tool for camera calibration using checkerboard pattern detection. Supports synthetic image generation, corner detection, intrinsic parameter estimation, distortion correction, and report export (PDF + TXT).

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- ReportLab (for PDF export)

```bash
pip install opencv-python numpy reportlab
```

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Interactive menu CLI |
| `calibrator.py` | Corner detection, calibration, undistortion |
| `img_generator.py` | Checkerboard pattern generator |
| `img_capture.py` | Synthetic image generator (perspective + distortion) |
| `reporter.py` | PDF and TXT report generation |
| `calib_images/` | Calibration images folder |
| `checkerboard.png` | Generated checkerboard pattern |

## Usage

Run the interactive menu:

```bash
python main.py
```

### Menu Options

1. **Generate & display checkerboard** — Create and preview a checkerboard pattern in a window
2. **Save checkerboard to file** — Generate and save the checkerboard as an image
3. **Auto-generate calibration images** — Create synthetic images with randomized perspective and lens distortion
4. **Preview sample variants** — Preview a few generated variants with simulated distortion
5. **Run calibration pipeline** — Detect corners and compute camera intrinsic matrix
6. **Undistort and rectify a test image** — Apply distortion correction to a test image
7. **Full pipeline** — End-to-end: generate → calibrate → report → undistort

### Programmatic Usage

```python
from calibrator import Calibrator

cal = Calibrator(
    image_dir="calib_images",
    pattern_size=(8, 5),   # inner corners: cols-1 x rows-1
    square_size=1.0,      # physical square size
)
cal.detect_corners()
cal.calibrate()
cal.undistort("calib_images/img_00.jpg", "undistorted.jpg")
```

## Output

After calibration, the following are generated:

- **`calibration_report.pdf`** — Full PDF report with camera matrix and distortion coefficients
- **`calibration_report.txt`** — Plain-text summary
- **`undistorted.jpg`** — Corrected/rectified test image

## Calibration Report (Sample)

```
Intrinsic Matrix (K):
  [fx,  0, cx]
  [ 0, fy, cy]
  [ 0,  0,  1]

Distortion Coefficients: k1, k2, p1, p2, k3
Reprojection Error: ~1.0 px (lower is better)
```

## How It Works

1. **Corner Detection** — Uses `cv2.findChessboardCorners` to locate inner checkerboard corners, refined to sub-pixel accuracy with `cv2.cornerSubPix`
2. **Calibration** — `cv2.calibrateCamera` computes the intrinsic matrix (K) and distortion coefficients
3. **Undistortion** — Applies `cv2.undistort` with the optimal new camera matrix to remove lens distortion
4. **Rectification** — Optionally warps the checkerboard to a fronto-parallel view via perspective transform
