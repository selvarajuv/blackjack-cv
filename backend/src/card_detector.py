import cv2
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from models import Card, DetectedCard, Rank, Suit


@dataclass
class CardTemplate:
    """Store reference images for each card"""

    rank: Rank
    suit: Suit
    image: np.ndarray


class CardDetector:
    def __init__(self):
        self.templates: List[CardTemplate] = []
        self.min_confidence = 0.75

        # Card detection parameters
        self.min_card_area = 1000
        self.card_aspect_ratio = 0.7  # width/height

    def detect_cards(self, frame: np.ndarray) -> List[DetectedCard]:
        """
        Main detection pipeline:
        1. Find card-shaped contours
        2. Extract and rectify card regions
        3. Match against templates or use OCR
        """
        detected_cards = []

        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Check if contour could be a card
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            width, height = rect[1]
            if width == 0 or height == 0:
                continue

            aspect_ratio = min(width, height) / max(width, height)
            area = width * height

            # Filter by size and shape
            if area > self.min_card_area and aspect_ratio > 0.5:
                # Extract card region
                card_region = self._extract_card_region(frame, box)
                if card_region is not None:
                    # Identify the card
                    card = self._identify_card(card_region)
                    if card:
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(box)
                        detected_cards.append(
                            DetectedCard(
                                card=card,
                                confidence=0.85,  # Placeholder
                                bbox=[x, y, w, h],
                            )
                        )

        return detected_cards

    def _extract_card_region(
        self, frame: np.ndarray, corners: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract and rectify card region to standard rectangle"""
        try:
            # Define target rectangle (standard card proportions)
            width, height = 200, 280
            dst_points = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
            )

            # Order corners consistently
            corners = self._order_corners(corners)

            # Get perspective transform
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
            warped = cv2.warpPerspective(frame, matrix, (width, height))

            return warped
        except Exception:
            return None

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum and diff to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def _identify_card(self, card_image: np.ndarray) -> Optional[Card]:
        """
        Identify card from image.
        For now, returns a dummy card - you'll implement:
        1. Template matching
        2. OCR on rank/suit
        3. Or ML model
        """
        # TODO: Implement actual card recognition
        # For testing, return a random card
        import random

        if random.random() > 0.2:  # 80% detection rate for testing
            rank = random.choice(list(Rank))
            suit = random.choice(list(Suit))
            return Card(rank=rank, suit=suit)
        return None

    def draw_detections(
        self, frame: np.ndarray, detections: List[DetectedCard]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        result = frame.copy()

        for detection in detections:
            x, y, w, h = detection.bbox
            card = detection.card

            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = (
                f"{card.rank.value}{card.suit.value[0]}"  # e.g., "AH" for Ace of Hearts
            )
            cv2.putText(
                result,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Draw confidence
            conf_text = f"{detection.confidence:.2f}"
            cv2.putText(
                result,
                conf_text,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )

        return result

    def calibrate_camera(self, frame: np.ndarray) -> bool:
        """Helper to set up camera view and lighting"""
        # Could add calibration logic here
        return True
