import cv2
import numpy as np
import os


class ImageCapture:
    """
    Automatically generates 15 synthetic calibration images from a
    checkerboard pattern by applying randomized perspective transforms,
    scaling, and positioning — no webcam or screen recording needed.

    Each generated image simulates the checkerboard being viewed from
    a different angle and distance, and includes synthetic lens distortion.
    This makes the generated images better for testing the calibration
    and undistort pipeline.

    Attributes:
        save_dir (str):      Directory where generated images will be saved.
        target (int):        Number of images to generate (default 15).
        output_size (tuple): Width and height of each output image in pixels.
    """

    def __init__(
        self,
        save_dir: str = "calib_images",
        target: int = 15,
        output_size: tuple = (1280, 720),
    ):
        self.save_dir    = save_dir
        self.target      = target
        self.output_size = output_size  # (width, height)

    def _ensure_dir(self) -> None:
        """Create the save directory if it does not already exist."""
        os.makedirs(self.save_dir, exist_ok=True)

    def _random_perspective(self, board: np.ndarray, index: int) -> np.ndarray:
        """
        Apply a randomized perspective warp + scale + translation to the
        checkerboard image to simulate a different camera viewing angle.

        The random seed is fixed per index so results are reproducible —
        running the program twice with the same settings gives the same images.

        Args:
            board (np.ndarray): The source checkerboard (grayscale).
            index (int):        Image index, used to seed the RNG.

        Returns:
            np.ndarray: BGR image of size self.output_size with the warped board.
        """
        rng = np.random.default_rng(seed=index * 42)   # reproducible per image

        h_src, w_src = board.shape[:2]
        W, H = self.output_size

        # --- Scale: fit the board into 40–75% of the output canvas ---
        scale = rng.uniform(0.40, 0.75)
        bw = int(w_src * scale)
        bh = int(h_src * scale)

        # --- Translation: place the scaled board randomly on the canvas ---
        max_tx = max(0, W - bw)
        max_ty = max(0, H - bh)
        tx = int(rng.uniform(0, max_tx))
        ty = int(rng.uniform(0, max_ty))

        # Source corners (before warp): the four corners of the scaled board
        src_pts = np.float32([
            [0,    0   ],
            [bw,   0   ],
            [bw,   bh  ],
            [0,    bh  ],
        ])

        # Destination corners (after warp): add random perspective offset
        # to each corner to simulate tilt/rotation
        perturb = bw * 0.22    # max offset = 22% of board width
        dst_pts = src_pts.copy()
        for i in range(4):
            dst_pts[i][0] += rng.uniform(-perturb, perturb) + tx
            dst_pts[i][1] += rng.uniform(-perturb * 0.6, perturb * 0.6) + ty

        # Clamp destination points to stay within the canvas
        dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, W - 1)
        dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, H - 1)

        # Resize board to scaled size first, then warp to output canvas
        board_scaled = cv2.resize(board, (bw, bh))
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(board_scaled, M, (W, H),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=128)   # gray background

        # Convert grayscale to BGR so downstream code can treat all images uniformly
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        return self._apply_lens_distortion(warped_bgr)

    def _apply_lens_distortion(
        self,
        image: np.ndarray,
        k1: float = -0.15,
        k2: float = 0.05,
        p1: float = 0.0,
        p2: float = 0.0,
        k3: float = 0.0,
    ) -> np.ndarray:
        """
        Apply a simple synthetic lens distortion model to the generated image.

        The distortion is applied in normalized image coordinates using the
        standard radial + tangential distortion equations, then remapped back
        to pixel coordinates.
        """
        h, w = image.shape[:2]
        fx = fy = max(1.0, min(w, h) * 0.8)
        cx = w / 2.0
        cy = h / 2.0

        xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        x = (xv - cx) / fx
        y = (yv - cy) / fy

        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
        x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        map_x = x_dist * fx + cx
        map_y = y_dist * fy + cy

        distorted = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=128,
        )
        return distorted

    def generate_variants(self, board: np.ndarray) -> int:
        """
        Generate self.target synthetic perspective variants of the checkerboard
        with added lens distortion, then save them directly to self.save_dir.

        Args:
            board (np.ndarray): Source checkerboard image (grayscale).

        Returns:
            int: Number of images successfully saved.
        """
        self._ensure_dir()
        print(f"[INFO] Generating {self.target} synthetic calibration images...")
        print(f"[INFO] Output size: {self.output_size[0]}x{self.output_size[1]} px")
        print(f"[INFO] Saving to: {os.path.abspath(self.save_dir)}")

        for i in range(self.target):
            img      = self._random_perspective(board, index=i)
            filename = os.path.join(self.save_dir, f"img_{i:02d}.jpg")
            cv2.imwrite(filename, img)
            print(f"[OK] {filename}")

        print(f"[DONE] {self.target} images saved to '{self.save_dir}'")
        return self.target

    def preview(self, board: np.ndarray, num_previews: int = 4) -> None:
        """
        Show a quick preview of the first `num_previews` generated variants
        in a single tiled window so the user can verify the output before
        proceeding to the calibration step.

        Press any key to close the preview window.

        Args:
            board (np.ndarray): Source checkerboard image (grayscale).
            num_previews (int): How many variants to tile in the preview.
        """
        previews = []
        for i in range(num_previews):
            img     = self._random_perspective(board, index=i)
            thumb   = cv2.resize(img, (320, 180))   # thumbnail size
            previews.append(thumb)

        # Tile thumbnails horizontally
        tiled = np.hstack(previews)
        cv2.imshow(f"Preview — {num_previews} sample variants (any key to close)", tiled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()