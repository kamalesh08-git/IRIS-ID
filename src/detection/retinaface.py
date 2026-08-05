"""
RetinaFace Face Detection Module for Masked Face Recognition System.

This module provides utilities to detect faces in images using the InsightFace
RetinaFace detector. It detects all faces, extracts facial landmarks, and
provides visualization capabilities.

Production-quality implementation with robust error handling and logging.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarks:
    """
    Dataclass to store facial landmarks for a detected face.

    Attributes:
        left_eye (Tuple[float, float]): Coordinates of left eye center.
        right_eye (Tuple[float, float]): Coordinates of right eye center.
        nose (Tuple[float, float]): Coordinates of nose tip.
        left_mouth (Tuple[float, float]): Coordinates of left mouth corner.
        right_mouth (Tuple[float, float]): Coordinates of right mouth corner.
    """
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    left_mouth: Tuple[float, float]
    right_mouth: Tuple[float, float]

    def to_dict(self) -> dict:
        """Convert landmarks to dictionary format."""
        return {
            'left_eye': self.left_eye,
            'right_eye': self.right_eye,
            'nose': self.nose,
            'left_mouth': self.left_mouth,
            'right_mouth': self.right_mouth
        }


@dataclass
class DetectedFace:
    """
    Dataclass to store information about a detected face.

    Attributes:
        bbox (Tuple[float, float, float, float]): Bounding box (x1, y1, x2, y2).
        confidence (float): Detection confidence score (0-1).
        landmarks (FaceLandmarks): Five facial landmarks.
        face_index (int): Index of this face in the detection results.
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    landmarks: FaceLandmarks
    face_index: int

    def __repr__(self) -> str:
        """Return readable representation of detected face."""
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        height = y2 - y1
        return (
            f"DetectedFace("
            f"index={self.face_index}, "
            f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
            f"size={width:.0f}x{height:.0f}, "
            f"confidence={self.confidence:.4f})"
        )


class RetinaFaceDetector:
    """
    Face detection using InsightFace RetinaFace detector.

    This class uses the state-of-the-art RetinaFace detector from InsightFace
    to detect faces and extract facial landmarks in images. Supports both
    local image files and numpy arrays as input.

    Attributes:
        detector (FaceAnalysis): InsightFace face detector instance.
        output_dir (Path): Directory for saving annotated images.
    """

    def __init__(self, output_dir: str = 'processed/detected') -> None:
        """
        Initialize the RetinaFace detector.

        Args:
            output_dir (str): Directory to save annotated images.
                             Default: 'processed/detected'

        Raises:
            RuntimeError: If detector initialization fails.
        """
        try:
            logger.info("Initializing RetinaFace detector...")

            # Initialize InsightFace with RetinaFace
            self.detector = FaceAnalysis(
                name='buffalo_l',  # High-quality model
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.detector.prepare(ctx_id=0, det_size=(640, 640))

            # Create output directory
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"RetinaFace detector initialized successfully")
            logger.info(f"Output directory: {self.output_dir.resolve()}")

        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise RuntimeError(f"Detector initialization failed: {e}")

    def _load_image(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Load image from file path or numpy array.

        Args:
            image_input (Union[str, Path, np.ndarray]): Either image file path
                                                       or numpy array.

        Returns:
            Optional[np.ndarray]: Loaded image in BGR format, or None if failed.
        """
        try:
            if isinstance(image_input, np.ndarray):
                # Image already loaded as numpy array
                logger.debug("Using provided numpy array")
                return image_input

            # Load from file path
            image_path = Path(image_input)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None

            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None

            logger.debug(f"Loaded image: {image_path.name} "
                        f"({image.shape[1]}x{image.shape[0]})")
            return image

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def _extract_landmarks(self, face_data: dict) -> Optional[FaceLandmarks]:
        """
        Extract five facial landmarks from detector output.

        Args:
            face_data (dict): Face data from InsightFace detector.

        Returns:
            Optional[FaceLandmarks]: Extracted landmarks or None if failed.
        """
        try:
            # InsightFace provides 5 landmarks: [left_eye, right_eye, nose,
            # left_mouth_corner, right_mouth_corner]
            landmarks = face_data.get('landmark_2d_106', 
                                     face_data.get('kps', None))

            if landmarks is None or len(landmarks) < 5:
                logger.warning("Invalid landmarks data")
                return None

            # Extract the 5 key landmarks
            # Note: If using 106 landmarks, we extract the main 5
            # If using kps (5 landmarks), use directly
            if len(landmarks) >= 5:
                left_eye = tuple(map(float, landmarks[0]))
                right_eye = tuple(map(float, landmarks[1]))
                nose = tuple(map(float, landmarks[2]))
                left_mouth = tuple(map(float, landmarks[3]))
                right_mouth = tuple(map(float, landmarks[4]))

                return FaceLandmarks(
                    left_eye=left_eye,
                    right_eye=right_eye,
                    nose=nose,
                    left_mouth=left_mouth,
                    right_mouth=right_mouth
                )

        except Exception as e:
            logger.warning(f"Error extracting landmarks: {e}")
            return None

    def detect(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], List[DetectedFace]]:
        """
        Detect all faces in an image.

        Args:
            image_input (Union[str, Path, np.ndarray]): Image file path or
                                                       numpy array.

        Returns:
            Tuple[Optional[np.ndarray], List[DetectedFace]]:
                - Original image (BGR format)
                - List of detected faces with bboxes and landmarks
        """
        logger.info(f"Starting face detection on image...")

        # Load image
        image = self._load_image(image_input)
        if image is None:
            logger.error("Failed to load image for detection")
            return None, []

        detected_faces: List[DetectedFace] = []

        try:
            # Run detection
            faces = self.detector.get(image)

            if not faces:
                logger.info("No faces detected")
                return image, detected_faces

            logger.info(f"Detected {len(faces)} face(s)")

            # Process each detected face
            for face_index, face in enumerate(faces):
                try:
                    # Extract bounding box
                    bbox = face.get('bbox', None)
                    if bbox is None or len(bbox) < 4:
                        logger.warning(f"Invalid bbox for face {face_index}")
                        continue

                    x1, y1, x2, y2 = bbox[:4]
                    bbox_tuple = (float(x1), float(y1), float(x2), float(y2))

                    # Extract confidence
                    confidence = float(face.get('det_score', 0.0))

                    # Extract landmarks
                    landmarks = self._extract_landmarks(face)
                    if landmarks is None:
                        logger.warning(
                            f"Could not extract landmarks for face {face_index}"
                        )
                        continue

                    # Create DetectedFace object
                    detected_face = DetectedFace(
                        bbox=bbox_tuple,
                        confidence=confidence,
                        landmarks=landmarks,
                        face_index=face_index
                    )

                    detected_faces.append(detected_face)
                    logger.debug(f"Processed face: {detected_face}")

                except Exception as e:
                    logger.warning(f"Error processing face {face_index}: {e}")
                    continue

            logger.info(f"Successfully detected {len(detected_faces)} faces")

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return image, []

        return image, detected_faces

    def _draw_detection(
        self, image: np.ndarray, detected_face: DetectedFace
    ) -> np.ndarray:
        """
        Draw bounding box and landmarks on image.

        Args:
            image (np.ndarray): Original image (BGR format).
            detected_face (DetectedFace): Detected face data.

        Returns:
            np.ndarray: Image with drawn annotations.
        """
        image_copy = image.copy()

        # Draw bounding box
        x1, y1, x2, y2 = detected_face.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Box color and thickness
        box_color = (0, 255, 0)  # Green in BGR
        box_thickness = 2

        cv2.rectangle(image_copy, (x1, y1), (x2, y2), box_color, box_thickness)

        # Draw confidence text
        confidence_text = f"Conf: {detected_face.confidence:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 255, 0)

        # Get text size for background
        text_size = cv2.getTextSize(
            confidence_text, font, font_scale, font_thickness
        )[0]
        text_x = x1
        text_y = y1 - 5

        # Draw background for text
        cv2.rectangle(
            image_copy,
            (text_x, text_y - text_size[1] - 4),
            (text_x + text_size[0] + 4, text_y + 4),
            (0, 0, 0),
            -1
        )

        # Draw text
        cv2.putText(
            image_copy,
            confidence_text,
            (text_x + 2, text_y - 2),
            font,
            font_scale,
            text_color,
            font_thickness
        )

        # Draw landmarks
        landmarks = detected_face.landmarks
        landmark_points = [
            landmarks.left_eye,
            landmarks.right_eye,
            landmarks.nose,
            landmarks.left_mouth,
            landmarks.right_mouth
        ]
        landmark_names = [
            'L_Eye', 'R_Eye', 'Nose', 'L_Mouth', 'R_Mouth'
        ]

        # Landmark circle properties
        landmark_radius = 4
        landmark_color = (0, 0, 255)  # Red in BGR
        landmark_thickness = -1  # Filled circle

        for point, name in zip(landmark_points, landmark_names):
            x, y = int(point[0]), int(point[1])

            # Draw circle
            cv2.circle(
                image_copy,
                (x, y),
                landmark_radius,
                landmark_color,
                landmark_thickness
            )

            # Draw label
            cv2.putText(
                image_copy,
                name,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1
            )

        return image_copy

    def visualize(
        self,
        image_input: Union[str, Path, np.ndarray],
        save: bool = True,
        display: bool = False
    ) -> Tuple[Optional[np.ndarray], List[DetectedFace]]:
        """
        Detect faces and visualize results.

        Args:
            image_input (Union[str, Path, np.ndarray]): Image file path or
                                                       numpy array.
            save (bool): Whether to save annotated image. Default: True
            display (bool): Whether to display image. Default: False

        Returns:
            Tuple[Optional[np.ndarray], List[DetectedFace]]:
                - Annotated image with drawings
                - List of detected faces
        """
        logger.info("Starting face detection and visualization...")

        # Detect faces
        image, detected_faces = self.detect(image_input)

        if image is None:
            logger.error("Image loading failed")
            return None, []

        if not detected_faces:
            logger.info("No faces to visualize")
            return image, detected_faces

        # Draw all detections on image
        annotated_image = image.copy()
        for detected_face in detected_faces:
            annotated_image = self._draw_detection(annotated_image, detected_face)

        # Save annotated image
        if save:
            self._save_annotated_image(image_input, annotated_image)

        # Display image
        if display:
            self._display_image(annotated_image, detected_faces)

        logger.info("Visualization complete")
        return annotated_image, detected_faces

    def _save_annotated_image(
        self, image_input: Union[str, Path, np.ndarray],
        annotated_image: np.ndarray
    ) -> bool:
        """
        Save annotated image to output directory.

        Args:
            image_input: Original image input.
            annotated_image (np.ndarray): Image with drawings.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        try:
            # Generate output filename
            if isinstance(image_input, np.ndarray):
                filename = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                image_path = Path(image_input)
                filename = f"detected_{image_path.stem}.jpg"

            output_path = self.output_dir / filename

            # Save image
            success = cv2.imwrite(str(output_path), annotated_image)

            if success:
                logger.info(f"Saved annotated image: {output_path}")
                return True
            else:
                logger.error(f"Failed to save image: {output_path}")
                return False

        except Exception as e:
            logger.error(f"Error saving annotated image: {e}")
            return False

    def _display_image(
        self, image: np.ndarray, detected_faces: List[DetectedFace]
    ) -> None:
        """
        Display image with detected faces.

        Args:
            image (np.ndarray): Annotated image.
            detected_faces (List[DetectedFace]): List of detections.
        """
        try:
            # Create window
            window_name = f"Face Detection - {len(detected_faces)} face(s) detected"
            cv2.imshow(window_name, image)

            logger.info("Displaying image. Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            logger.warning(f"Could not display image: {e}")

    def get_detection_summary(
        self, detected_faces: List[DetectedFace]
    ) -> dict:
        """
        Get summary statistics of detected faces.

        Args:
            detected_faces (List[DetectedFace]): List of detections.

        Returns:
            dict: Summary statistics.
        """
        if not detected_faces:
            return {
                'total_faces': 0,
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'bbox_sizes': []
            }

        confidences = [f.confidence for f in detected_faces]
        sizes = []

        for face in detected_faces:
            x1, y1, x2, y2 = face.bbox
            width = x2 - x1
            height = y2 - y1
            sizes.append((width, height))

        return {
            'total_faces': len(detected_faces),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'bbox_sizes': sizes
        }


def main() -> None:
    """
    Demo main function showing RetinaFaceDetector usage.

    This demonstrates:
    1. Detector initialization
    2. Face detection on sample image
    3. Result visualization
    4. Statistics reporting
    """
    logger.info("Starting RetinaFace Detector Demo")

    try:
        # Initialize detector
        detector = RetinaFaceDetector(output_dir='processed/detected')

        # Demo 1: Detect on sample image
        # You can replace with your own image path
        sample_image_path = 'sample_image.jpg'

        logger.info(f"Demo: Detecting faces in sample image...")

        # Detect and visualize
        annotated_image, detected_faces = detector.visualize(
            image_input=sample_image_path,
            save=True,
            display=False  # Set to True to show image
        )

        if annotated_image is not None:
            # Print results
            print("\n" + "="*60)
            print("Face Detection Results")
            print("="*60)

            print(f"\nTotal faces detected: {len(detected_faces)}")

            for face in detected_faces:
                print(f"\n{face}")
                print(f"  Bounding Box: {face.bbox}")
                print(f"  Confidence: {face.confidence:.4f}")
                print(f"  Landmarks:")
                print(f"    Left Eye:   {face.landmarks.left_eye}")
                print(f"    Right Eye:  {face.landmarks.right_eye}")
                print(f"    Nose:       {face.landmarks.nose}")
                print(f"    Left Mouth: {face.landmarks.left_mouth}")
                print(f"    Right Mouth: {face.landmarks.right_mouth}")

            # Get summary
            summary = detector.get_detection_summary(detected_faces)
            print(f"\n" + "="*60)
            print("Detection Summary")
            print("="*60)
            print(f"Total Faces: {summary['total_faces']}")
            print(f"Avg Confidence: {summary['avg_confidence']:.4f}")
            print(f"Min Confidence: {summary['min_confidence']:.4f}")
            print(f"Max Confidence: {summary['max_confidence']:.4f}")

            if summary['bbox_sizes']:
                avg_width = np.mean([s[0] for s in summary['bbox_sizes']])
                avg_height = np.mean([s[1] for s in summary['bbox_sizes']])
                print(f"Avg Face Size: {avg_width:.0f}x{avg_height:.0f}")

            print("="*60 + "\n")

        else:
            logger.warning("No image was processed")

    except FileNotFoundError:
        logger.warning("Sample image not found. Please provide a test image.")
        logger.info("Usage: Place an image file and specify its path in the code.")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
