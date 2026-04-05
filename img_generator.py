import cv2
import numpy as np
import os


class CheckerboardGenerator:
    """
    Generates a black-and-white checkerboard pattern image
    suitable for use in camera calibration pipelines.

    Attributes:
        rows (int):        Number of square rows in the checkerboard.
        cols (int):        Number of square columns in the checkerboard.
        square_size (int): Side length of each square in pixels.
        board (np.ndarray | None): The generated image; None until generate() is called.
    """

    def __init__(self, rows: int = 6, cols: int = 9, square_size: int = 80):
        self.rows        = rows
        self.cols        = cols
        self.square_size = square_size
        self.board       = None

    def generate(self) -> np.ndarray:
        """
        Build the checkerboard image and store it in self.board.

        Cells where (row + col) is even are filled white (255);
        others remain black (0), producing the alternating pattern.

        Returns:
            np.ndarray: Grayscale image of shape (rows*square_size, cols*square_size).
        """
        h = self.rows * self.square_size
        w = self.cols * self.square_size
        board = np.zeros((h, w), dtype=np.uint8)

        for r in range(self.rows):
            for c in range(self.cols):
                if (r + c) % 2 == 0:
                    r0, r1 = r * self.square_size, (r + 1) * self.square_size
                    c0, c1 = c * self.square_size, (c + 1) * self.square_size
                    board[r0:r1, c0:c1] = 255

        self.board = board
        return board

    def save(self, path: str = "checkerboard.png") -> None:
        """
        Save the checkerboard image to disk.
        Calls generate() automatically if not yet generated.

        Args:
            path: Destination file path (default "checkerboard.png").
        """
        if self.board is None:
            self.generate()
        cv2.imwrite(path, self.board)
        print(f"[OK] Checkerboard saved to: {os.path.abspath(path)}")

    def show(self) -> None:
        """
        Display the checkerboard in an OpenCV window.
        The window stays open until the user presses any key.
        Calls generate() automatically if not yet generated.
        """
        if self.board is None:
            self.generate()
        print("[INFO] Displaying checkerboard. Press any key to close the window.")
        cv2.imshow("Checkerboard Pattern", self.board)
        cv2.waitKey(0)
        cv2.destroyAllWindows()