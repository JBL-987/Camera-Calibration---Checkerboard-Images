import os
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


class Reporter:
    """
    Generates a formatted PDF calibration report from the results
    produced by Calibrator.calibrate().

    The report includes:
        - Camera intrinsic matrix K (focal lengths, principal point)
        - Distortion coefficients (k1, k2, p1, p2, k3)
        - Mean reprojection error + quality assessment
        - Timestamp and configuration summary

    Usage:
        reporter = Reporter(calibrator)
        reporter.save_pdf("calibration_report.pdf")
        reporter.print_report()   # terminal summary
    """

    # Color palette
    COLOR_PRIMARY   = colors.HexColor("#1a1a2e")   # dark navy — headings
    COLOR_ACCENT    = colors.HexColor("#16213e")   # section bars
    COLOR_HIGHLIGHT = colors.HexColor("#0f3460")   # table header
    COLOR_LIGHT     = colors.HexColor("#e8f4f8")   # table row alt
    COLOR_WHITE     = colors.white
    COLOR_GOOD      = colors.HexColor("#2ecc71")
    COLOR_FAIR      = colors.HexColor("#f39c12")
    COLOR_POOR      = colors.HexColor("#e74c3c")

    def __init__(self, calibrator):
        """
        Args:
            calibrator: A Calibrator instance after calibrate() has been called.
        """
        self.cal    = calibrator
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    # ------------------------------------------------------------------ #
    #  Style setup
    # ------------------------------------------------------------------ #

    def _setup_styles(self) -> None:
        """Register custom paragraph styles used throughout the PDF."""
        self.style_title = ParagraphStyle(
            "ReportTitle",
            parent    = self.styles["Title"],
            fontSize  = 22,
            textColor = self.COLOR_PRIMARY,
            spaceAfter= 4,
            alignment = TA_CENTER,
            fontName  = "Helvetica-Bold",
        )
        self.style_subtitle = ParagraphStyle(
            "ReportSubtitle",
            parent    = self.styles["Normal"],
            fontSize  = 10,
            textColor = colors.HexColor("#555555"),
            spaceAfter= 2,
            alignment = TA_CENTER,
        )
        self.style_section = ParagraphStyle(
            "SectionHeader",
            parent    = self.styles["Heading2"],
            fontSize  = 12,
            textColor = self.COLOR_WHITE,
            backColor = self.COLOR_ACCENT,
            spaceBefore = 14,
            spaceAfter  = 6,
            leftIndent  = 8,
            fontName    = "Helvetica-Bold",
        )
        self.style_body = ParagraphStyle(
            "Body",
            parent    = self.styles["Normal"],
            fontSize  = 10,
            textColor = colors.HexColor("#222222"),
            spaceAfter= 4,
            leading   = 14,
        )
        self.style_mono = ParagraphStyle(
            "Mono",
            parent    = self.styles["Code"],
            fontSize  = 9,
            fontName  = "Courier",
            textColor = colors.HexColor("#1a1a2e"),
            spaceAfter= 2,
        )
        self.style_quality = ParagraphStyle(
            "Quality",
            parent    = self.styles["Normal"],
            fontSize  = 13,
            fontName  = "Helvetica-Bold",
            alignment = TA_CENTER,
            spaceAfter= 6,
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _quality_info(self, error: float) -> tuple:
        """
        Return (label, color) based on reprojection error.

        Thresholds:
            < 0.5 px  → Excellent
            < 1.0 px  → Good
            < 2.0 px  → Fair
            >= 2.0 px → Poor
        """
        if error < 0.5:
            return "Excellent", self.COLOR_GOOD
        elif error < 1.0:
            return "Good", self.COLOR_GOOD
        elif error < 2.0:
            return "Fair — consider recalibrating", self.COLOR_FAIR
        else:
            return "Poor — recalibrate with better images", self.COLOR_POOR

    def _table_style(self, header_rows: int = 1) -> TableStyle:
        """Return a standard TableStyle for data tables."""
        return TableStyle([
            # Header row
            ("BACKGROUND",  (0, 0), (-1, header_rows - 1), self.COLOR_HIGHLIGHT),
            ("TEXTCOLOR",   (0, 0), (-1, header_rows - 1), self.COLOR_WHITE),
            ("FONTNAME",    (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, header_rows - 1), 10),
            ("ALIGN",       (0, 0), (-1, header_rows - 1), "CENTER"),
            # Body rows — alternating background
            ("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
             [self.COLOR_WHITE, self.COLOR_LIGHT]),
            ("FONTNAME",    (0, header_rows), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, header_rows), (-1, -1), 9),
            ("ALIGN",       (1, header_rows), (-1, -1), "CENTER"),
            ("ALIGN",       (0, header_rows), (0, -1),  "LEFT"),
            # Grid
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0,0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",(0, 0), (-1, -1), 8),
        ])

    # ------------------------------------------------------------------ #
    #  PDF section builders
    # ------------------------------------------------------------------ #

    def _section_summary(self) -> list:
        """Build the configuration summary section."""
        cal  = self.cal
        now  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        rows = [
            ["Parameter",       "Value"],
            ["Generated",       now],
            ["Image directory", cal.image_dir],
            ["Images used",     str(len(cal.image_points))],
            ["Pattern size",    f"{cal.pattern_size[0]} x {cal.pattern_size[1]} inner corners"],
            ["Image size",      f"{cal.image_size[0]} x {cal.image_size[1]} px"],
            ["Square size",     f"{cal.square_size}"],
        ]
        tbl = Table(rows, colWidths=[5 * cm, 10 * cm])
        tbl.setStyle(self._table_style(header_rows=1))
        return [
            Paragraph("Configuration Summary", self.style_section),
            tbl,
        ]

    def _section_reprojection(self) -> list:
        """Build the reprojection error section."""
        err   = self.cal.reprojection_error
        label, color = self._quality_info(err)

        quality_para = Paragraph(
            f'<font color="{color.hexval()}">{label}</font>',
            self.style_quality,
        )

        rows = [
            ["Metric",                  "Value"],
            ["Mean reprojection error", f"{err:.4f} px"],
            ["Quality assessment",      label],
        ]
        tbl = Table(rows, colWidths=[7 * cm, 8 * cm])
        tbl.setStyle(self._table_style(header_rows=1))

        note = Paragraph(
            "<b>Interpretation:</b>  &lt; 0.5 px = Excellent &nbsp;|&nbsp; "
            "0.5–1.0 px = Good &nbsp;|&nbsp; 1.0–2.0 px = Fair &nbsp;|&nbsp; "
            "&gt; 2.0 px = Poor",
            self.style_body,
        )
        return [
            Paragraph("Reprojection Error", self.style_section),
            quality_para,
            tbl,
            Spacer(1, 6),
            note,
        ]

    def _section_intrinsics(self) -> list:
        """Build the camera intrinsic matrix K section."""
        K  = self.cal.K
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # K matrix as a formatted table (3 rows × 3 cols)
        k_data = [
            ["K matrix",  "Col 0",          "Col 1",          "Col 2"],
            ["Row 0",     f"{K[0,0]:.4f}",  f"{K[0,1]:.4f}", f"{K[0,2]:.4f}"],
            ["Row 1",     f"{K[1,0]:.4f}",  f"{K[1,1]:.4f}", f"{K[1,2]:.4f}"],
            ["Row 2",     f"{K[2,0]:.4f}",  f"{K[2,1]:.4f}", f"{K[2,2]:.4f}"],
        ]
        k_tbl = Table(k_data, colWidths=[3 * cm, 4 * cm, 4 * cm, 4 * cm])
        k_tbl.setStyle(self._table_style(header_rows=1))

        # Intrinsic parameters breakdown
        param_data = [
            ["Parameter", "Symbol", "Value",       "Unit"],
            ["Focal length X", "fx", f"{fx:.4f}", "px"],
            ["Focal length Y", "fy", f"{fy:.4f}", "px"],
            ["Principal point X", "cx", f"{cx:.4f}", "px"],
            ["Principal point Y", "cy", f"{cy:.4f}", "px"],
        ]
        p_tbl = Table(param_data, colWidths=[5.5*cm, 2.5*cm, 4*cm, 3*cm])
        p_tbl.setStyle(self._table_style(header_rows=1))

        return [
            Paragraph("Intrinsic Matrix (K)", self.style_section),
            k_tbl,
            Spacer(1, 10),
            Paragraph("Intrinsic Parameters", self.style_section),
            p_tbl,
        ]

    def _section_distortion(self) -> list:
        """Build the distortion coefficients section."""
        dist = self.cal.dist.flatten()
        k1   = dist[0]
        k2   = dist[1]
        p1   = dist[2]
        p2   = dist[3]
        k3   = dist[4] if len(dist) > 4 else 0.0

        rows = [
            ["Coefficient", "Symbol", "Value",       "Type"],
            ["Radial 1",    "k1",     f"{k1:+.6f}",  "Radial"],
            ["Radial 2",    "k2",     f"{k2:+.6f}",  "Radial"],
            ["Tangential 1","p1",     f"{p1:+.6f}",  "Tangential"],
            ["Tangential 2","p2",     f"{p2:+.6f}",  "Tangential"],
            ["Radial 3",    "k3",     f"{k3:+.6f}",  "Radial"],
        ]
        tbl = Table(rows, colWidths=[5*cm, 2.5*cm, 4.5*cm, 3*cm])
        tbl.setStyle(self._table_style(header_rows=1))

        note = Paragraph(
            "<b>Note:</b> Radial distortion (k1, k2, k3) causes barrel or "
            "pincushion warping. Tangential distortion (p1, p2) occurs when "
            "the lens is not perfectly parallel to the image sensor.",
            self.style_body,
        )
        return [
            Paragraph("Distortion Coefficients", self.style_section),
            tbl,
            Spacer(1, 6),
            note,
        ]

    # ------------------------------------------------------------------ #
    #  Public methods
    # ------------------------------------------------------------------ #

    def save_pdf(self, path: str = "calibration_report.pdf") -> None:
        """
        Build and save the full calibration report as a PDF.

        Args:
            path (str): Output file path (default "calibration_report.pdf").
        """
        if self.cal.K is None:
            print("[ERROR] No calibration data. Run Calibrator.calibrate() first.")
            return

        doc = SimpleDocTemplate(
            path,
            pagesize    = A4,
            leftMargin  = 2   * cm,
            rightMargin = 2   * cm,
            topMargin   = 2   * cm,
            bottomMargin= 2   * cm,
        )

        story = []

        # --- Title block ---
        story.append(Paragraph("Camera Calibration Report", self.style_title))
        story.append(Paragraph("OpenCV Checkerboard Calibration Pipeline", self.style_subtitle))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=self.COLOR_PRIMARY, spaceAfter=12))

        # --- Sections ---
        story += self._section_summary()
        story.append(Spacer(1, 8))
        story += self._section_reprojection()
        story.append(Spacer(1, 8))
        story += self._section_intrinsics()
        story.append(Spacer(1, 8))
        story += self._section_distortion()

        # --- Footer note ---
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#aaaaaa")))
        story.append(Paragraph(
            "Generated by Camera Calibration Tool  •  OpenCV + ReportLab",
            self.style_subtitle,
        ))

        doc.build(story)
        print(f"[OK] PDF report saved to: {os.path.abspath(path)}")

    def save_txt(self, path: str = "calibration_report.txt") -> None:
        """
        Save a plain-text version of the report (fallback / quick reference).

        Args:
            path (str): Output file path.
        """
        if self.cal.K is None:
            print("[ERROR] No calibration data.")
            return

        cal  = self.cal
        K    = cal.K
        dist = cal.dist.flatten()
        err  = cal.reprojection_error
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label, _ = self._quality_info(err)

        lines = [
            "=" * 55,
            "  CAMERA CALIBRATION REPORT",
            "=" * 55,
            f"  Generated : {now}",
            f"  Image dir : {cal.image_dir}",
            f"  Images    : {len(cal.image_points)}",
            f"  Pattern   : {cal.pattern_size[0]}x{cal.pattern_size[1]}",
            f"  Img size  : {cal.image_size[0]}x{cal.image_size[1]} px",
            "",
            "--- Intrinsic Matrix (K) ---",
            f"  [ {K[0,0]:10.4f}  {K[0,1]:10.4f}  {K[0,2]:10.4f} ]",
            f"  [ {K[1,0]:10.4f}  {K[1,1]:10.4f}  {K[1,2]:10.4f} ]",
            f"  [ {K[2,0]:10.4f}  {K[2,1]:10.4f}  {K[2,2]:10.4f} ]",
            "",
            f"  fx : {K[0,0]:.4f} px",
            f"  fy : {K[1,1]:.4f} px",
            f"  cx : {K[0,2]:.4f} px",
            f"  cy : {K[1,2]:.4f} px",
            "",
            "--- Distortion Coefficients ---",
            f"  k1 : {dist[0]:+.6f}",
            f"  k2 : {dist[1]:+.6f}",
            f"  p1 : {dist[2]:+.6f}",
            f"  p2 : {dist[3]:+.6f}",
            f"  k3 : {dist[4] if len(dist) > 4 else 0.0:+.6f}",
            "",
            "--- Reprojection Error ---",
            f"  Mean error : {err:.4f} px",
            f"  Quality    : {label}",
            "=" * 55,
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[OK] TXT report saved to: {os.path.abspath(path)}")

    def print_report(self) -> None:
        """Print a compact summary to the terminal."""
        if self.cal.K is None:
            print("[ERROR] No calibration data.")
            return
        K    = self.cal.K
        dist = self.cal.dist.flatten()
        err  = self.cal.reprojection_error
        label, _ = self._quality_info(err)

        print("\n" + "=" * 45)
        print("  CALIBRATION RESULT SUMMARY")
        print("=" * 45)
        print(f"  fx = {K[0,0]:.4f} px   fy = {K[1,1]:.4f} px")
        print(f"  cx = {K[0,2]:.4f} px   cy = {K[1,2]:.4f} px")
        print(f"  k1 = {dist[0]:+.6f}    k2 = {dist[1]:+.6f}")
        print(f"  p1 = {dist[2]:+.6f}    p2 = {dist[3]:+.6f}")
        print(f"  Reprojection error : {err:.4f} px  [{label}]")
        print("=" * 45)