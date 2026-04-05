import cv2
import numpy as np
import os
import glob


class Calibrator:
    """
    Runs the full camera calibration pipeline on a folder of checkerboard images.

    Pipeline:
        1. Load all images from the calibration folder.
        2. Detect checkerboard corners in each image (cv2.findChessboardCorners).
        3. Refine corner positions to sub-pixel accuracy (cv2.cornerSubPix).
        4. Run cv2.calibrateCamera to compute K, distortion coefficients, R, t.
        5. Compute mean reprojection error across all images.
        6. Undistort a test image using the computed parameters.

    Attributes:
        image_dir (str):    Folder containing the calibration images.
        pattern_size (tuple): Inner corner count as (cols-1, rows-1).
                              For a 9x6 board: (8, 5) inner corners.
        square_size (float): Physical size of each square (e.g. 1.0 = unit, or mm).
        K (np.ndarray):     Camera intrinsic matrix (3x3), set after calibrate().
        dist (np.ndarray):  Distortion coefficients (1x5), set after calibrate().
        rvecs (list):       Rotation vectors per image, set after calibrate().
        tvecs (list):       Translation vectors per image, set after calibrate().
        reprojection_error (float): Mean reprojection error, set after calibrate().
        image_points (list): Detected 2D corner points per image.
        object_points (list): Corresponding 3D object points per image.
        image_size (tuple): (width, height) of the calibration images.
    """

    def __init__(
        self,
        image_dir: str = "calib_images",
        pattern_size: tuple = (8, 5),
        square_size: float = 1.0,
    ):
        self.image_dir   = image_dir
        self.pattern_size = pattern_size   # (inner_cols, inner_rows)
        self.square_size  = square_size

        # Results — populated after calibrate() is called
        self.K                  = None
        self.dist               = None
        self.rvecs              = None
        self.tvecs              = None
        self.reprojection_error = None
        self.image_points       = []
        self.object_points      = []
        self.image_size         = None

        # Sub-pixel refinement termination criteria
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,    # max iterations
            0.001, # epsilon (convergence threshold)
        )

    def _build_object_points(self) -> np.ndarray:
        """
        Build the 3D object point grid for one checkerboard image.

        Object points are the idealized 3D positions of the inner corners
        on the flat checkerboard plane (z=0), spaced by square_size.

        Returns:
            np.ndarray: Shape (pattern_size[0]*pattern_size[1], 3), dtype float32.
        """
        cols, rows = self.pattern_size
        objp = np.zeros((rows * cols, 3), dtype=np.float32)

        # Fill X, Y columns; Z stays 0 (flat plane)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def detect_corners(self) -> int:
        """
        Load all images from self.image_dir and detect checkerboard corners.

        For each image where corners are found, the detected 2D points are
        refined to sub-pixel accuracy and stored in self.image_points.
        The corresponding 3D object points are stored in self.object_points.

        Returns:
            int: Number of images where corners were successfully detected.
        """
        pattern   = os.path.join(self.image_dir, "*.jpg")
        filenames = sorted(glob.glob(pattern))

        if not filenames:
            print(f"[ERROR] No .jpg images found in '{self.image_dir}'")
            return 0

        print(f"[INFO] Found {len(filenames)} images in '{self.image_dir}'")
        objp = self._build_object_points()
        success_count = 0

        for fpath in filenames:
            img  = cv2.imread(fpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Store image size from the first image (all must be the same size)
            if self.image_size is None:
                self.image_size = (gray.shape[1], gray.shape[0])  # (width, height)

            # --- Step 2: detect inner corners ---
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            if found:
                # --- Step 3: refine to sub-pixel accuracy ---
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self._criteria
                )
                self.image_points.append(corners_refined)
                self.object_points.append(objp)
                success_count += 1
                print(f"  [OK]   {os.path.basename(fpath)}")
            else:
                print(f"  [SKIP] {os.path.basename(fpath)} — corners not found")

        print(f"[INFO] Corners detected in {success_count}/{len(filenames)} images")
        return success_count

    def calibrate(self) -> bool:
        """
        Run cv2.calibrateCamera using the detected corner points.

        Must call detect_corners() first. Populates self.K, self.dist,
        self.rvecs, self.tvecs, and self.reprojection_error.

        Returns:
            bool: True if calibration succeeded, False otherwise.
        """
        if not self.image_points:
            print("[ERROR] No corner points available. Run detect_corners() first.")
            return False

        print("[INFO] Running calibration...")
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,  # initial camera matrix (None = auto)
            None,  # initial distortion coefficients (None = auto)
        )

        self.K     = K
        self.dist  = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        # --- Compute mean reprojection error ---
        total_error = 0.0
        for i, objp in enumerate(self.object_points):
            # Project 3D points back to 2D using the calibrated parameters
            projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
            # Compute Euclidean distance between detected and reprojected points
            error = cv2.norm(self.image_points[i], projected, cv2.NORM_L2)
            total_error += error / len(projected)

        self.reprojection_error = total_error / len(self.object_points)
        print(f"[OK] Calibration complete.")
        print(f"[OK] Mean reprojection error: {self.reprojection_error:.4f} px")
        return True

    def _rectify_checkerboard(self, image: np.ndarray) -> np.ndarray | None:
        """
        Warp the detected checkerboard region to a fronto-parallel view.

        Returns the rectified image when the inner corner grid can be detected.
        If the checkerboard pattern cannot be found, returns None.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        if not found:
            return None

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), self._criteria
        )

        cols, rows = self.pattern_size
        if cols < 2 or rows < 2:
            return None

        top_left = corners_refined[0, 0]
        top_right = corners_refined[cols - 1, 0]
        bottom_left = corners_refined[-cols, 0]
        bottom_right = corners_refined[-1, 0]

        width = int(np.linalg.norm(top_right - top_left))
        height = int(np.linalg.norm(bottom_left - top_left))
        if width <= 1 or height <= 1:
            return None

        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        dst = np.array([
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        rectified = cv2.warpPerspective(
            image,
            H,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=128,
        )
        return rectified

    def undistort(self, image_path: str, output_path: str = "undistorted.jpg") -> bool:
        """
        Apply the calibrated distortion correction to a test image and save it.

        Uses cv2.getOptimalNewCameraMatrix to crop out the black border regions
        that appear after undistortion (alpha=0 means fully cropped). If the
        checkerboard is detected, the undistorted image is also rectified to a
        fronto-parallel view.

        Args:
            image_path (str):  Path to the distorted test image.
            output_path (str): Where to save the undistorted result.

        Returns:
            bool: True if undistortion succeeded, False otherwise.
        """
        if self.K is None or self.dist is None:
            print("[ERROR] Camera not calibrated. Run calibrate() first.")
            return False

        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return False

        h, w = img.shape[:2]

        # Compute the optimal new camera matrix that removes black borders
        # alpha=0: crop to valid pixels only; alpha=1: keep all pixels
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (w, h), alpha=0
        )

        # Apply undistortion
        undistorted = cv2.undistort(img, self.K, self.dist, None, new_K)

        # Crop to the valid region of interest
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y+rh, x:x+rw]

        rectified = self._rectify_checkerboard(undistorted)
        if rectified is not None:
            undistorted = rectified
            print("[OK] Perspective rectification applied.")
        else:
            print("[WARN] Checkerboard not detected for rectification. Saved undistorted-only image.")

        cv2.imwrite(output_path, undistorted)
        print(f"[OK] Undistorted image saved to: {os.path.abspath(output_path)}")
        return True