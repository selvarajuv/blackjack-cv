import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import csv


@dataclass
class TestConditions:
    """Structure for test conditions"""

    lighting_type: str  # artificial_overhead, artificial_desk
    lighting_intensity: str  # medium
    distance_inches: int  # 6, 12, 18
    angle_degrees: int  # 0, 30
    card_rank: str  # A
    card_suit: str  # spades
    card_brand: str  # bicycle
    card_condition: str  # new
    background_type: str  # dark_table
    movement_type: str  # static, steady_movement


@dataclass
class AlgorithmParams:
    """Algorithm parameters to test"""

    bkg_thresh_offset: int = (
        60  # Increased from 40 for better overhead lighting performance
    )
    card_min_area: int = 5000
    card_max_area: int = 200000
    epsilon_factor: float = 0.02
    detail_threshold: int = 120
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 0.85


@dataclass
class TestResults:
    """Results from a single test"""

    test_id: str
    conditions: TestConditions
    params: AlgorithmParams
    total_frames: int
    frames_with_detection: int
    total_detections: int
    false_positives: int
    detection_rate: float
    false_positive_rate: float
    avg_processing_time: float
    corner_quality_score: float  # 1-5 subjective
    consistency_score: float  # variance in detection
    notes: str
    timestamp: str


class CardDetectionTester:
    def __init__(self):
        self.results_dir = "test_results"
        self.videos_dir = "test_videos"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        # Current test parameters
        self.params = AlgorithmParams()

        # Quality assessment
        self.corner_quality_samples = []

        # Debug mode state
        self.debug_mode = False
        self.paused = True
        self.current_frame_idx = 0
        self.total_frames = 0

    def debug_video_analysis(self, video_path: str):
        """Interactive debug mode for video analysis"""
        print("\n=== DEBUG MODE ACTIVATED ===")
        print(f"Analyzing: {os.path.basename(video_path)}")
        print("\nControls:")
        print("SPACE: Play/Pause")
        print("←→: Previous/Next frame (when paused)")
        print("1-5: Change visualization mode")
        print("+/-: Adjust threshold offset")
        print("r: Reset to start of video")
        print("q: Quit debug mode")
        print("=" * 40)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0

        print(f"Video: {self.total_frames} frames at {fps:.1f} FPS")

        # Start at first frame
        ret, frame = cap.read()
        if not ret:
            print("Cannot read first frame")
            cap.release()
            return

        current_frame = frame
        self.current_frame_idx = 1
        mode = 5  # Start with detection overlay

        # Statistics tracking for debug mode - only update when processing new frames
        debug_stats = {
            "frames_processed": 0,
            "frames_with_detection": 0,
            "total_detections": 0,
            "current_detection_rate": 0.0,
        }

        # Track which frames we've already counted to avoid double-counting
        processed_frames = set()

        while True:
            # Generate analysis view
            result_frame, detection_info = self._create_debug_analysis_view(
                current_frame, mode
            )

            # Only update statistics for new frames (avoid counting same frame multiple times)
            if self.current_frame_idx not in processed_frames:
                debug_stats["frames_processed"] += 1
                if detection_info["num_detections"] > 0:
                    debug_stats["frames_with_detection"] += 1
                    debug_stats["total_detections"] += detection_info["num_detections"]

                debug_stats["current_detection_rate"] = (
                    debug_stats["frames_with_detection"]
                    / debug_stats["frames_processed"]
                    * 100
                    if debug_stats["frames_processed"] > 0
                    else 0
                )
                processed_frames.add(self.current_frame_idx)

            # Add comprehensive frame info overlay
            info_lines = [
                f"Frame: {self.current_frame_idx}/{self.total_frames}",
                f"Threshold Offset: {self.params.bkg_thresh_offset}",
                f"Detection Rate: {debug_stats['current_detection_rate']:.1f}%",
                f"Current Frame Detections: {detection_info['num_detections']}",
                f"Background Level: {detection_info['bkg_level']}",
                f"Threshold Level: {detection_info['thresh_level']}",
            ]

            y_pos = 25
            for line in info_lines:
                cv2.putText(
                    result_frame,
                    line,
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_pos += 25

            # Show current mode
            mode_names = {
                1: "Original",
                2: "Threshold",
                3: "Contours",
                4: "Side-by-Side",
                5: "Detection Overlay",
            }
            cv2.putText(
                result_frame,
                f"Mode: {mode_names.get(mode, 'Unknown')}",
                (10, result_frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Debug Analysis - Testing Framework", result_frame)

            # Handle input
            wait_time = max(1, int(1000 / fps)) if not self.paused else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):  # Space - play/pause
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "PLAYING"
                print(f"{status} - Frame {self.current_frame_idx}/{self.total_frames}")

            elif key == 83 and self.paused:  # Right arrow - next frame
                if self.current_frame_idx < self.total_frames:
                    ret, frame = cap.read()
                    if ret:
                        current_frame = frame
                        self.current_frame_idx += 1
                    else:
                        print("End of video reached")

            elif key == 81 and self.paused:  # Left arrow - previous frame
                if self.current_frame_idx > 1:
                    # Go back by seeking to the desired frame
                    target_frame = max(0, self.current_frame_idx - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ret, frame = cap.read()
                    if ret:
                        current_frame = frame
                        self.current_frame_idx = target_frame + 1
                    else:
                        print("Cannot go to previous frame")

            elif key == ord("1"):
                mode = 1
                print("Mode: Original")
            elif key == ord("2"):
                mode = 2
                print("Mode: Threshold Analysis")
            elif key == ord("3"):
                mode = 3
                print("Mode: Contour Analysis")
            elif key == ord("4"):
                mode = 4
                print("Mode: Side-by-Side")
            elif key == ord("5"):
                mode = 5
                print("Mode: Detection Overlay")

            elif key == ord("+") or key == ord("="):
                self.params.bkg_thresh_offset += 10
                print(f"Threshold offset increased to: {self.params.bkg_thresh_offset}")

            elif key == ord("-"):
                self.params.bkg_thresh_offset = max(
                    10, self.params.bkg_thresh_offset - 10
                )
                print(f"Threshold offset decreased to: {self.params.bkg_thresh_offset}")

            elif key == ord("r"):
                # Reset to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    current_frame = frame
                    self.current_frame_idx = 1
                    debug_stats = {
                        "frames_processed": 0,
                        "frames_with_detection": 0,
                        "total_detections": 0,
                        "current_detection_rate": 0.0,
                    }
                    processed_frames = set()
                print("Reset to start of video")

            # Continue playing if not paused (NO FRAME SKIPPING)
            if not self.paused:
                ret, frame = cap.read()
                if ret:
                    current_frame = frame
                    self.current_frame_idx += 1
                else:
                    print("End of video reached")
                    self.paused = True

        cap.release()
        cv2.destroyAllWindows()

        # Print final debug statistics
        print(f"\nDebug Session Summary:")
        print(f"Frames Analyzed: {debug_stats['frames_processed']}")
        print(f"Detection Rate: {debug_stats['current_detection_rate']:.1f}%")
        print(f"Total Detections: {debug_stats['total_detections']}")

    def _create_debug_analysis_view(self, frame, mode):
        """Create analysis view and return detection info"""
        # Run detection to get info
        boundaries, debug_info = self._detect_cards(frame)

        detection_info = {
            "num_detections": len(boundaries),
            "bkg_level": debug_info["bkg_level"],
            "thresh_level": debug_info["thresh_level"],
        }

        if mode == 1:
            return self._debug_original_view(frame, detection_info), detection_info
        elif mode == 2:
            return self._debug_threshold_analysis(frame), detection_info
        elif mode == 3:
            return self._debug_contour_analysis(
                frame, boundaries, debug_info
            ), detection_info
        elif mode == 4:
            return self._debug_side_by_side_analysis(frame), detection_info
        elif mode == 5:
            return self._debug_detection_overlay(
                frame, boundaries, debug_info
            ), detection_info

        return frame, detection_info

    def _debug_original_view(self, frame, detection_info):
        """Debug version of original view"""
        max_width, max_height = 800, 600
        h, w = frame.shape[:2]

        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            result = cv2.resize(frame, (new_w, new_h))
        else:
            result = frame.copy()

        # Add detection status
        status_text = f"DETECTIONS: {detection_info['num_detections']}"
        status_color = (
            (0, 255, 0) if detection_info["num_detections"] > 0 else (0, 0, 255)
        )

        cv2.putText(
            result,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        return result

    def _debug_threshold_analysis(self, frame):
        """Debug threshold analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background sampling (same as detection algorithm)
        img_h, img_w = gray.shape
        bkg_samples = [
            gray[int(img_h / 10)][int(img_w / 2)],
            gray[int(img_h / 2)][int(img_w / 10)],
            gray[int(img_h / 2)][int(img_w * 0.9)],
        ]
        bkg_level = int(np.mean(bkg_samples))
        thresh_level = bkg_level + self.params.bkg_thresh_offset

        # Apply threshold
        _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

        # Resize for screen
        max_width, max_height = 800, 600
        if thresh.shape[1] > max_width or thresh.shape[0] > max_height:
            scale = min(max_width / thresh.shape[1], max_height / thresh.shape[0])
            new_w, new_h = int(thresh.shape[1] * scale), int(thresh.shape[0] * scale)
            thresh = cv2.resize(thresh, (new_w, new_h))

        # Convert to color for display
        result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Add analysis info
        cv2.putText(
            result,
            f"Background: {bkg_level}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result,
            f"Threshold: {thresh_level}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Calculate white pixel percentage
        white_pixels = np.sum(thresh == 255)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        white_ratio = white_pixels / total_pixels * 100

        cv2.putText(
            result,
            f"White: {white_ratio:.1f}%",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return result

    def _debug_contour_analysis(self, frame, boundaries, debug_info):
        """Debug contour analysis with detailed feedback"""
        max_width, max_height = 800, 600
        h, w = frame.shape[:2]

        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            result = cv2.resize(frame, (new_w, new_h))
            scale_factor = scale
        else:
            result = frame.copy()
            scale_factor = 1.0

        # Get threshold image to find all contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, debug_info["thresh_level"], 255, cv2.THRESH_BINARY
        )
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Show all significant contours with analysis
        if contours:
            index_sort = sorted(
                range(len(contours)),
                key=lambda i: cv2.contourArea(contours[i]),
                reverse=True,
            )

            for idx, i in enumerate(index_sort[:8]):  # Show top 8 contours
                contour = contours[i]
                area = cv2.contourArea(contour)

                if area < 1000:  # Skip tiny contours
                    continue

                # Analyze this contour
                peri = cv2.arcLength(contour, True)
                approx = (
                    cv2.approxPolyDP(contour, self.params.epsilon_factor * peri, True)
                    if peri > 0
                    else []
                )

                # Get aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = (
                    min(width, height) / max(width, height)
                    if width > 0 and height > 0
                    else 0
                )

                # Check hierarchy
                has_parent = hierarchy is not None and hierarchy[0][i][3] != -1

                # Determine issues
                issues = []
                if area < self.params.card_min_area:
                    issues.append("SMALL")
                if area > self.params.card_max_area:
                    issues.append("BIG")
                if len(approx) != 4:
                    issues.append(f"{len(approx)}CORN")
                if has_parent:
                    issues.append("PARENT")
                if (
                    aspect_ratio < self.params.aspect_ratio_min
                    or aspect_ratio > self.params.aspect_ratio_max
                ):
                    issues.append(f"ASPECT")

                # Color code based on validation
                if len(issues) == 0:
                    color = (0, 255, 0)  # Green - valid card
                elif len(approx) == 4:
                    color = (0, 255, 255)  # Yellow - rectangular but other issues
                else:
                    color = (0, 0, 255)  # Red - not rectangular

                # Scale and draw contour
                if scale_factor != 1.0:
                    scaled_contour = (contour * scale_factor).astype(np.int32)
                else:
                    scaled_contour = contour

                cv2.drawContours(result, [scaled_contour], -1, color, 2)

                # Add label
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if scale_factor != 1.0:
                    x, y = int(x * scale_factor), int(y * scale_factor)

                label = f"#{idx + 1}: {area:.0f}"
                if issues:
                    label += f" ({issues[0]})"

                cv2.putText(
                    result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        # Add summary info
        cv2.putText(
            result,
            f"Valid Cards: {len(boundaries)}",
            (10, result.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result,
            f"Total Contours: {len(contours)}",
            (10, result.shape[0] - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return result

    def _debug_side_by_side_analysis(self, frame):
        """Debug side-by-side view"""
        max_width = 1200
        max_height = 600

        h, w = frame.shape[:2]
        aspect_ratio = w / h

        if aspect_ratio > 1:
            panel_width = min(max_width // 2, 600)
            panel_height = int(panel_width / aspect_ratio)
        else:
            panel_height = min(max_height, 600)
            panel_width = int(panel_height * aspect_ratio)

        panel_width = max(300, min(panel_width, 600))
        panel_height = max(200, min(panel_height, 600))

        # Original frame
        original = cv2.resize(frame, (panel_width, panel_height))

        # Threshold analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        img_h, img_w = gray.shape
        bkg_level = int(
            np.mean(
                [
                    gray[int(img_h / 10)][int(img_w / 2)],
                    gray[int(img_h / 2)][int(img_w / 10)],
                    gray[int(img_h / 2)][int(img_w * 0.9)],
                ]
            )
        )
        thresh_level = bkg_level + self.params.bkg_thresh_offset

        _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_resized = cv2.resize(thresh_bgr, (panel_width, panel_height))

        # Combine side by side
        result = np.hstack([original, thresh_resized])

        # Add labels
        font_scale = 0.6 if panel_width < 500 else 0.8
        cv2.putText(
            result,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result,
            "Threshold",
            (panel_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
        )

        # Add threshold info
        cv2.putText(
            result,
            f"BG:{bkg_level} T:{thresh_level}",
            (panel_width + 10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.8,
            (255, 255, 255),
            1,
        )

        return result

    def _debug_detection_overlay(self, frame, boundaries, debug_info):
        """Debug detection overlay - same as VisualFailureAnalyzer but using testing framework algorithm"""
        max_width, max_height = 900, 700
        h, w = frame.shape[:2]

        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            result = cv2.resize(frame, (new_w, new_h))
            scale_factor = scale
        else:
            result = frame.copy()
            scale_factor = 1.0

        # Draw detected card boundaries with detailed info
        if boundaries:
            for i, boundary in enumerate(boundaries):
                contour = boundary["contour"]

                # Scale contour if frame was resized
                if scale_factor != 1.0:
                    scaled_contour = (contour * scale_factor).astype(np.int32)
                else:
                    scaled_contour = contour

                # Draw green rectangle for detected cards
                cv2.drawContours(result, [scaled_contour], -1, (0, 255, 0), 3)

                # Draw corner points
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(
                    contour, self.params.epsilon_factor * peri, True
                )

                for point in approx:
                    if scale_factor != 1.0:
                        scaled_point = (
                            int(point[0][0] * scale_factor),
                            int(point[0][1] * scale_factor),
                        )
                    else:
                        scaled_point = tuple(point[0])
                    cv2.circle(result, scaled_point, 8, (255, 0, 0), -1)

                # Add card info
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if scale_factor != 1.0:
                    x, y = int(x * scale_factor), int(y * scale_factor)

                cv2.putText(
                    result,
                    f"CARD {i + 1}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    result,
                    f"Area: {boundary['area']:.0f}",
                    (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    result,
                    f"Aspect: {boundary['aspect_ratio']:.2f}",
                    (x, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Detection status banner
        if boundaries:
            status_color = (0, 255, 0)
            status_text = f"✓ DETECTING {len(boundaries)} CARD(S)"
        else:
            status_color = (0, 0, 255)
            status_text = "✗ NO CARDS DETECTED"

        # Add black background for text visibility
        cv2.rectangle(result, (5, 5), (400, 35), (0, 0, 0), -1)
        cv2.putText(
            result,
            status_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        return result

    def generate_test_matrix(self) -> List[TestConditions]:
        """Generate focused 12-test matrix"""
        test_matrix = []

        # Focused variables
        lighting_types = ["artificial_overhead", "artificial_desk"]
        lighting_intensities = ["medium"]  # Focus on most common condition
        distances = [16, 24, 32]  # inches - close, medium, far
        angles = [0, 30]  # degrees - straight and angled
        cards = [("A", "spades")]  # Single card is sufficient for shape detection
        backgrounds = ["dark_table"]  # Most common blackjack table
        movements = ["static", "steady_movement"]  # Core movement types

        test_id = 1
        for lighting_type in lighting_types:
            for intensity in lighting_intensities:
                for distance in distances:
                    for angle in angles:
                        for card_rank, card_suit in cards:
                            for movement in movements:
                                conditions = TestConditions(
                                    lighting_type=lighting_type,
                                    lighting_intensity=intensity,
                                    distance_inches=distance,
                                    angle_degrees=angle,
                                    card_rank=card_rank,
                                    card_suit=card_suit,
                                    card_brand="bicycle",
                                    card_condition="new",
                                    background_type="dark_table",
                                    movement_type=movement,
                                )
                                test_matrix.append(conditions)
                                test_id += 1

        return test_matrix

    def generate_test_protocol(self, output_file: str = "test_protocol.txt"):
        """Generate human-readable test protocol"""
        test_matrix = self.generate_test_matrix()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("FOCUSED TEST MATRIX - ESSENTIAL VARIABLES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total tests: {len(test_matrix)} (streamlined for efficiency)\n")
            f.write(
                f"Estimated time: {len(test_matrix) * 1.5:.0f} minutes (1.5 min per test)\n\n"
            )

            f.write("VARIABLES TESTED:\n")
            f.write("* Lighting: Overhead, Desk lamp\n")
            f.write("* Distance: 16in (close), 24in (medium), 32in (far)\n")
            f.write("* Angle: 0° (straight), 30° (angled)\n")
            f.write("* Card: Ace of Spades (single card test)\n")
            f.write("* Movement: Static, Steady movement\n")
            f.write("* Background: Dark table (standard blackjack)\n\n")

            f.write("EQUIPMENT NEEDED:\n")
            f.write("- Camera/webcam\n")
            f.write("- Playing card (Ace of Spades, Bicycle brand preferred)\n")
            f.write("- Dark table surface\n")
            f.write("- Desk lamp (adjustable brightness)\n")
            f.write('- Ruler for distance measurement (16", 24", 32")\n')
            f.write("- Angle guide (protractor app or 30° reference)\n\n")

            f.write("VIDEO RECORDING INSTRUCTIONS:\n")
            f.write("- Record 15-20 seconds per test\n")
            f.write("- Keep card centered in frame\n")
            f.write("- For 'static': hold card steady\n")
            f.write("- For 'steady_movement': slow, smooth movements\n")
            f.write("- Use naming: test_001.MOV, test_002.MOV, etc.\n")
            f.write("- Save all videos in test_videos/ folder\n\n")

            f.write("TEST CONDITIONS:\n")
            f.write("-" * 30 + "\n")

            for i, test in enumerate(test_matrix, 1):
                f.write(f"\nTest {i:03d}:\n")
                f.write(f"  Video: test_{i:03d}.MOV\n")
                f.write(f"  Card: {test.card_rank} of {test.card_suit}\n")
                f.write(f"  Distance: {test.distance_inches}in from camera\n")
                f.write(f"  Angle: {test.angle_degrees}° tilt\n")
                f.write(f"  Lighting: {test.lighting_type}\n")
                f.write(f"  Movement: {test.movement_type}\n")

            f.write(f"\nTOTAL: {len(test_matrix)} tests")

        print(f"Test protocol saved to {output_file}")
        return len(test_matrix)

    def analyze_video(self, video_path: str, conditions: TestConditions) -> TestResults:
        """Analyze a single test video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Results tracking
        frames_with_detection = 0
        total_detections = 0
        false_positives = 0
        processing_times = []
        detection_consistency = []  # Track detection count per frame

        print(f"Analyzing {video_path} ({total_frames} frames)...")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame with timing
            start_time = time.time()
            boundaries, debug_info = self._detect_cards(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Count detections
            num_detections = len(boundaries)
            detection_consistency.append(num_detections)

            if num_detections > 0:
                frames_with_detection += 1
                total_detections += num_detections

                # Assess corner quality (sample every 10th frame)
                if frame_count % 10 == 0:
                    quality = self._assess_corner_quality(frame, boundaries)
                    self.corner_quality_samples.append(quality)

            # Simple false positive detection (more than 2 cards detected)
            if num_detections > 2:
                false_positives += num_detections - 2

        cap.release()

        # Calculate metrics
        detection_rate = (
            (frames_with_detection / total_frames) * 100 if total_frames > 0 else 0
        )
        false_positive_rate = (
            (false_positives / total_frames) * 100 if total_frames > 0 else 0
        )
        avg_processing_time = np.mean(processing_times) if processing_times else 0

        # Consistency score (lower variance = higher consistency)
        consistency_variance = (
            np.var(detection_consistency) if detection_consistency else 0
        )
        consistency_score = max(
            0, 5 - consistency_variance
        )  # 5 = perfect, 0 = very inconsistent

        # Corner quality score
        corner_quality_score = (
            np.mean(self.corner_quality_samples) if self.corner_quality_samples else 0
        )

        # Generate test ID
        test_id = f"{conditions.lighting_type}_{conditions.distance_inches}in_{conditions.angle_degrees}deg_{conditions.movement_type}"

        results = TestResults(
            test_id=test_id,
            conditions=conditions,
            params=self.params,
            total_frames=total_frames,
            frames_with_detection=frames_with_detection,
            total_detections=total_detections,
            false_positives=false_positives,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            avg_processing_time=avg_processing_time,
            corner_quality_score=corner_quality_score,
            consistency_score=consistency_score,
            notes="",
            timestamp=datetime.now().isoformat(),
        )

        return results

    def _detect_cards(self, frame):
        """EXACT SAME card detection algorithm as used in testing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background sampling - EXACT SAME as testing
        img_h, img_w = gray.shape
        bkg_samples = [
            gray[int(img_h / 10)][int(img_w / 2)],
            gray[int(img_h / 2)][int(img_w / 10)],
            gray[int(img_h / 2)][int(img_w * 0.9)],
        ]
        bkg_level = int(np.mean(bkg_samples))
        thresh_level = bkg_level + self.params.bkg_thresh_offset

        # Threshold - EXACT SAME
        _, boundary_thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

        # Find contours - EXACT SAME
        contours, hierarchy = cv2.findContours(
            boundary_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        boundaries = []
        debug_info = {
            "total_contours": len(contours),
            "bkg_level": bkg_level,
            "thresh_level": thresh_level,
        }

        if len(contours) > 0:
            # Sort by area - EXACT SAME
            index_sort = sorted(
                range(len(contours)),
                key=lambda i: cv2.contourArea(contours[i]),
                reverse=True,
            )

            for i in index_sort[:5]:  # Check top 5 contours - EXACT SAME
                contour = contours[i]
                area = cv2.contourArea(contour)

                if area < self.params.card_min_area or area > self.params.card_max_area:
                    continue

                peri = cv2.arcLength(contour, True)
                if peri == 0:
                    continue

                approx = cv2.approxPolyDP(
                    contour, self.params.epsilon_factor * peri, True
                )

                if len(approx) != 4:
                    continue

                # Aspect ratio check - EXACT SAME
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = min(width, height) / max(width, height)
                    if not (
                        self.params.aspect_ratio_min
                        <= aspect_ratio
                        <= self.params.aspect_ratio_max
                    ):
                        continue

                # Hierarchy check - EXACT SAME
                if hierarchy is not None and hierarchy[0][i][3] != -1:
                    continue

                boundaries.append(
                    {"contour": contour, "area": area, "aspect_ratio": aspect_ratio}
                )

        return boundaries, debug_info

    def _assess_corner_quality(self, frame, boundaries) -> float:
        """Assess corner extraction quality (1-5 scale)"""
        if not boundaries:
            return 0

        total_quality = 0

        for boundary in boundaries:
            area = boundary["area"]
            aspect = boundary["aspect_ratio"]

            # Larger cards with good aspect ratios get higher scores
            area_score = min(5, area / 10000)  # Normalize area
            aspect_score = 5 if 0.6 <= aspect <= 0.8 else 3  # Ideal playing card ratio

            card_quality = (area_score + aspect_score) / 2
            total_quality += card_quality

        return total_quality / len(boundaries)

    def run_test_suite(self, videos_directory: str = None):
        """Run complete test suite on all videos"""
        videos_dir = videos_directory or self.videos_dir

        if not os.path.exists(videos_dir):
            print(f"Videos directory not found: {videos_dir}")
            return

        video_files = [
            f
            for f in os.listdir(videos_dir)
            if f.lower().endswith((".mov", ".mp4", ".avi"))
        ]

        if not video_files:
            print(f"No video files found in {videos_dir}")
            return

        print(f"Found {len(video_files)} test videos")

        all_results = []

        for video_file in sorted(video_files):
            video_path = os.path.join(videos_dir, video_file)

            # Extract test conditions from filename
            conditions = self._parse_video_filename(video_file)

            try:
                results = self.analyze_video(video_path, conditions)
                all_results.append(results)

                print(f"✓ {video_file}: {results.detection_rate:.1f}% detection rate")

            except Exception as e:
                print(f"✗ {video_file}: Error - {e}")

        # Save results
        self._save_test_results(all_results)

        # Generate report
        self._generate_summary_report(all_results)

        return all_results

    def debug_specific_video(self, video_filename: str):
        """Debug a specific video with interactive visualization"""
        video_path = os.path.join(self.videos_dir, video_filename)

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")

            # List available videos
            video_files = [
                f
                for f in os.listdir(self.videos_dir)
                if f.lower().endswith((".mov", ".mp4", ".avi"))
            ]
            if video_files:
                print(f"\nAvailable videos in {self.videos_dir}/:")
                for i, video in enumerate(sorted(video_files), 1):
                    print(f"  {i}. {video}")
            return

        print(f"\n🔍 DEBUG MODE: {video_filename}")
        print("This uses the EXACT SAME algorithm as the testing framework")
        print(
            "Any differences you see here vs. your visual analyzer indicate algorithm discrepancies"
        )

        self.debug_video_analysis(video_path)

    def _parse_video_filename(self, filename: str) -> TestConditions:
        """Parse test conditions from video filename"""
        import re

        match = re.search(r"test_(\d+)", filename.lower())

        if match:
            test_num = int(match.group(1))
            test_matrix = self.generate_test_matrix()

            if 1 <= test_num <= len(test_matrix):
                return test_matrix[test_num - 1]

        # Default if parsing fails
        return TestConditions(
            lighting_type="unknown",
            lighting_intensity="medium",
            distance_inches=16,
            angle_degrees=0,
            card_rank="A",
            card_suit="spades",
            card_brand="bicycle",
            card_condition="new",
            background_type="dark_table",
            movement_type="unknown",
        )

    def _save_test_results(self, results: List[TestResults]):
        """Save test results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed JSON
        json_file = f"{self.results_dir}/test_results_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Save CSV summary
        csv_file = f"{self.results_dir}/test_summary_{timestamp}.csv"
        with open(csv_file, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))

        print(f"Results saved: {json_file}, {csv_file}")

    def _generate_summary_report(self, results: List[TestResults]):
        """Generate human-readable summary report"""
        if not results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.results_dir}/summary_report_{timestamp}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("CARD DETECTION TEST RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(results)}\n\n")

            # Overall statistics
            detection_rates = [r.detection_rate for r in results]
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  Average Detection Rate: {np.mean(detection_rates):.1f}%\n")
            f.write(f"  Best Detection Rate: {np.max(detection_rates):.1f}%\n")
            f.write(f"  Worst Detection Rate: {np.min(detection_rates):.1f}%\n")
            f.write(f"  Standard Deviation: {np.std(detection_rates):.1f}%\n\n")

            # Tests above/below thresholds
            excellent_tests = [r for r in results if r.detection_rate >= 90]
            good_tests = [r for r in results if 70 <= r.detection_rate < 90]
            poor_tests = [r for r in results if r.detection_rate < 70]

            f.write("PERFORMANCE BREAKDOWN:\n")
            f.write(f"  Excellent (>=90%): {len(excellent_tests)} tests\n")
            f.write(f"  Good (70-89%): {len(good_tests)} tests\n")
            f.write(f"  Poor (<70%): {len(poor_tests)} tests\n\n")

            # Best and worst performing tests
            f.write("BEST PERFORMING TESTS:\n")
            best_tests = sorted(results, key=lambda x: x.detection_rate, reverse=True)[
                :5
            ]
            for i, test in enumerate(best_tests, 1):
                f.write(f"  {i}. {test.test_id}: {test.detection_rate:.1f}%\n")

            f.write("\nWORST PERFORMING TESTS:\n")
            worst_tests = sorted(results, key=lambda x: x.detection_rate)[:5]
            for i, test in enumerate(worst_tests, 1):
                f.write(f"  {i}. {test.test_id}: {test.detection_rate:.1f}%\n")

            f.write(f"\nRECOMMENDATIONS:\n")
            if np.mean(detection_rates) >= 85:
                f.write(
                    "* System performance is excellent. Ready for template matching.\n"
                )
            elif np.mean(detection_rates) >= 70:
                f.write("! System performance is good but needs improvement.\n")
                f.write("  Consider parameter optimization or algorithm refinements.\n")
            else:
                f.write("X System performance needs significant improvement.\n")
                f.write("  Review algorithm parameters and test conditions.\n")

        print(f"Summary report saved: {report_file}")


def main():
    import sys

    tester = CardDetectionTester()

    if len(sys.argv) > 1 and sys.argv[1] == "--generate-protocol":
        num_tests = tester.generate_test_protocol()
        print(f"Generated focused protocol with {num_tests} tests")

    elif len(sys.argv) > 1 and sys.argv[1] == "--run-tests":
        print("Running focused test suite...")
        results = tester.run_test_suite()
        print(f"Completed {len(results) if results else 0} tests")

    elif len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # Debug mode for specific video
        if len(sys.argv) > 2:
            video_filename = sys.argv[2]
        else:
            video_filename = input(
                "Enter video filename (e.g., test_004.MOV): "
            ).strip()

        tester.debug_specific_video(video_filename)

    else:
        print("ENHANCED CARD DETECTION TESTING FRAMEWORK")
        print("=" * 50)
        print("Commands:")
        print("  --generate-protocol    Generate focused 12-test protocol")
        print("  --run-tests           Run all tests in test_videos/")
        print("  --debug [filename]    Interactive debug mode for specific video")
        print()
        print("New Debug Feature:")
        print("  python testing_framework.py --debug test_004.MOV")
        print("  (Uses EXACT same algorithm as automated testing)")
        print()
        print("Workflow:")
        print("1. python testing_framework.py --generate-protocol")
        print("2. Record 12 videos following protocol")
        print("3. python testing_framework.py --run-tests")
        print("4. python testing_framework.py --debug [problem_video]")


if __name__ == "__main__":
    main()
