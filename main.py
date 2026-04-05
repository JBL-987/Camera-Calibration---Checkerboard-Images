from img_generator import CheckerboardGenerator
from img_capture import ImageCapture
from calibrator import Calibrator
from reporter import Reporter


def print_menu():
    """Display the main menu options to the user."""
    print("\n" + "=" * 45)
    print("   CAMERA CALIBRATION TOOL")
    print("=" * 45)
    print("  [1] Generate & display checkerboard")
    print("  [2] Save checkerboard to file")
    print("  [3] Auto-generate calibration images")
    print("  [4] Preview sample variants")
    print("      (includes synthetic lens distortion)")
    print("  [5] Run calibration pipeline")
    print("  [6] Undistort and rectify a test image")
    print("  [7] Full pipeline (generate → calibrate → report)")
    print("  [0] Exit")
    print("=" * 45)


def menu_generate_show():
    """Generate the checkerboard pattern and display it in a window."""
    rows = int(input("  Number of rows [default 6]: ") or 6)
    cols = int(input("  Number of cols [default 9]: ") or 9)
    size = int(input("  Square size in px [default 80]: ") or 80)
    gen  = CheckerboardGenerator(rows=rows, cols=cols, square_size=size)
    gen.generate()
    gen.show()


def menu_save():
    """Generate the checkerboard and save it to an image file."""
    rows = int(input("  Number of rows [default 6]: ") or 6)
    cols = int(input("  Number of cols [default 9]: ") or 9)
    size = int(input("  Square size in px [default 80]: ") or 80)
    path = input("  Output filename [default checkerboard.png]: ") or "checkerboard.png"
    gen  = CheckerboardGenerator(rows=rows, cols=cols, square_size=size)
    gen.generate()
    gen.save(path)


def menu_auto_generate():
    """Auto-generate synthetic calibration images with randomized perspective and lens distortion."""
    folder = input("  Save folder [default calib_images]: ") or "calib_images"
    target = int(input("  Number of images [default 15]: ") or 15)
    width  = int(input("  Output width px [default 1280]: ") or 1280)
    height = int(input("  Output height px [default 720]: ") or 720)
    gen    = CheckerboardGenerator()
    gen.generate()
    cap = ImageCapture(save_dir=folder, target=target, output_size=(width, height))
    cap.generate_variants(gen.board)


def menu_preview():
    """Show a tiled preview of a few sample variants."""
    rows = int(input("  Number of rows [default 6]: ") or 6)
    cols = int(input("  Number of cols [default 9]: ") or 9)
    size = int(input("  Square size in px [default 80]: ") or 80)
    num  = int(input("  Number of previews to show [default 4]: ") or 4)
    gen  = CheckerboardGenerator(rows=rows, cols=cols, square_size=size)
    gen.generate()
    cap = ImageCapture()
    cap.preview(gen.board, num_previews=num)


def _run_calibration(folder, pat_cols, pat_rows, sq_size) -> Calibrator:
    """
    Internal helper: run corner detection + calibration and return
    the populated Calibrator instance.
    """
    cal = Calibrator(
        image_dir    = folder,
        pattern_size = (pat_cols, pat_rows),
        square_size  = sq_size,
    )
    found = cal.detect_corners()
    if found == 0:
        print("[ERROR] No corners detected. Check images and pattern_size.")
        return cal
    cal.calibrate()
    return cal


def _save_reports(cal: Calibrator) -> None:
    """
    Internal helper: prompt for output paths and save both PDF and TXT reports.
    """
    rep = Reporter(cal)
    rep.print_report()

    pdf_path = input("  PDF report path [default calibration_report.pdf]: ") \
               or "calibration_report.pdf"
    txt_path = input("  TXT report path [default calibration_report.txt]: ") \
               or "calibration_report.txt"

    rep.save_pdf(pdf_path)
    rep.save_txt(txt_path)


def menu_calibrate() -> Calibrator:
    """Run calibration pipeline, save PDF + TXT reports, return Calibrator."""
    folder   = input("  Calibration images folder [default calib_images]: ") or "calib_images"
    pat_cols = int(input("  Inner corners — cols [default 8]: ") or 8)
    pat_rows = int(input("  Inner corners — rows [default 5]: ") or 5)
    sq_size  = float(input("  Square size (mm or 1.0 for unit) [default 1.0]: ") or 1.0)

    cal = _run_calibration(folder, pat_cols, pat_rows, sq_size)
    if cal.K is not None:
        _save_reports(cal)
    return cal


def menu_undistort(cal: Calibrator = None) -> None:
    """Undistort and rectify a test image. Runs calibration first if not already done."""
    if cal is None or cal.K is None:
        print("[INFO] No calibration loaded. Running calibration first...")
        cal = menu_calibrate()
        if cal.K is None:
            return

    test_img = input("  Path to test image [default calib_images/img_00.jpg]: ") \
               or "calib_images/img_00.jpg"
    out_path = input("  Output path [default undistorted.jpg]: ") \
               or "undistorted.jpg"
    cal.undistort(test_img, out_path)


def menu_full_pipeline():
    """
    End-to-end pipeline:
      [1/5] Generate checkerboard
      [2/5] Auto-generate calibration images
      [3/5] Detect corners & calibrate
      [4/5] Save PDF + TXT report
      [5/5] Undistort test image
    """
    # 1 — Generate board
    print("\n[1/5] Generating checkerboard...")
    rows = int(input("  Number of rows [default 6]: ") or 6)
    cols = int(input("  Number of cols [default 9]: ") or 9)
    size = int(input("  Square size in px [default 80]: ") or 80)
    gen  = CheckerboardGenerator(rows=rows, cols=cols, square_size=size)
    gen.generate()
    gen.save()

    # 2 — Auto-generate images
    print("\n[2/5] Auto-generating calibration images...")
    folder = input("  Save folder [default calib_images]: ") or "calib_images"
    target = int(input("  Number of images [default 15]: ") or 15)
    width  = int(input("  Output width px [default 1280]: ") or 1280)
    height = int(input("  Output height px [default 720]: ") or 720)
    cap    = ImageCapture(save_dir=folder, target=target, output_size=(width, height))
    cap.generate_variants(gen.board)

    # 3 — Calibrate
    print("\n[3/5] Detecting corners & calibrating...")
    cal = _run_calibration(
        folder   = folder,
        pat_cols = cols - 1,   # inner corners = board cols - 1
        pat_rows = rows - 1,
        sq_size  = 1.0,
    )
    if cal.K is None:
        print("[ERROR] Calibration failed. Stopping pipeline.")
        return

    # 4 — Report
    print("\n[4/5] Saving reports...")
    rep = Reporter(cal)
    rep.print_report()
    rep.save_pdf("calibration_report.pdf")
    rep.save_txt("calibration_report.txt")

    # 5 — Undistort
    print("\n[5/5] Undistorting test image...")
    cal.undistort(
        image_path  = f"{folder}/img_00.jpg",
        output_path = "undistorted.jpg",
    )

    print("\n[DONE] Full pipeline complete. Output files:")
    print("  checkerboard.png          — base pattern")
    print(f"  {folder}/                 — calibration images")
    print("  calibration_report.pdf    — full PDF report")
    print("  calibration_report.txt    — plain-text report")
    print("  undistorted.jpg           — corrected test image")


def main():
    """
    Main entry point. Runs an infinite loop displaying the menu
    and dispatching to the appropriate function based on user input.
    """
    last_calibrator = None   # reuse across menu_calibrate → menu_undistort

    while True:
        print_menu()
        choice = input("  Select option: ").strip()

        if choice == "1":
            menu_generate_show()
        elif choice == "2":
            menu_save()
        elif choice == "3":
            menu_auto_generate()
        elif choice == "4":
            menu_preview()
        elif choice == "5":
            last_calibrator = menu_calibrate()
        elif choice == "6":
            menu_undistort(last_calibrator)
        elif choice == "7":
            menu_full_pipeline()
        elif choice == "0":
            print("\n[BYE] Program exited.\n")
            break
        else:
            print("[!] Invalid option, please try again.")


if __name__ == "__main__":
    main()