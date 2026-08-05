"""
Example Usage Patterns for RetinaFaceDetector

This script demonstrates various ways to use the RetinaFaceDetector module
for your masked face recognition system.
"""

import logging
from pathlib import Path
import cv2
import numpy as np
from retinaface import RetinaFaceDetector, DetectedFace


def example_1_basic_detection() -> None:
    """Example 1: Basic face detection on a single image."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Face Detection")
    print("="*60)

    try:
        detector = RetinaFaceDetector(output_dir='processed/detected')

        # You need to provide your own image
        image_path = 'test_image.jpg'

        print(f"Detecting faces in: {image_path}")

        image, faces = detector.detect(image_path)

        if image is not None:
            print(f"\n✓ Detection complete")
            print(f"  Found {len(faces)} face(s)")

            for i, face in enumerate(faces):
                print(f"\n  Face {i+1}:")
                print(f"    Bounding Box: {face.bbox}")
                print(f"    Confidence: {face.confidence:.4f}")

        else:
            print("✗ Failed to load image")

    except Exception as e:
        print(f"Note: {e}")
        print("Please provide a test image file.")


def example_2_visualization() -> None:
    """Example 2: Detect and visualize with annotations."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Visualization with Annotations")
    print("="*60)

    try:
        detector = RetinaFaceDetector(output_dir='processed/detected')

        image_path = 'test_image.jpg'

        print(f"Processing: {image_path}")

        # Detect and visualize
        annotated, faces = detector.visualize(
            image_input=image_path,
            save=True,
            display=False  # Set to True to show image window
        )

        if annotated is not None:
            print(f"✓ Visualization complete")
            print(f"  Detected: {len(faces)} faces")
            print(f"  Annotated image saved to: processed/detected/")

    except Exception as e:
        print(f"Note: {e}")


def example_3_landmark_extraction() -> None:
    """Example 3: Extract and display facial landmarks."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Facial Landmark Extraction")
    print("="*60)

    try:
        detector = RetinaFaceDetector()

        image_path = 'test_image.jpg'
        image, faces = detector.detect(image_path)

        if image is not None and faces:
            print(f"\nExtracted landmarks for {len(faces)} face(s):\n")

            for i, face in enumerate(faces):
                landmarks = face.landmarks

                print(f"Face {i+1} Landmarks:")
                print(f"  Left Eye:    {landmarks.left_eye}")
                print(f"  Right Eye:   {landmarks.right_eye}")
                print(f"  Nose:        {landmarks.nose}")
                print(f"  Left Mouth:  {landmarks.left_mouth}")
                print(f"  Right Mouth: {landmarks.right_mouth}")
                print()

        else:
            print("No faces detected or image not found")

    except Exception as e:
        print(f"Note: {e}")


def example_4_confidence_filtering() -> None:
    """Example 4: Filter detections by confidence threshold."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Confidence Filtering")
    print("="*60)

    try:
        detector = RetinaFaceDetector()

        image_path = 'test_image.jpg'
        image, all_faces = detector.detect(image_path)

        if image is not None:
            # Filter by confidence
            thresholds = [0.9, 0.95, 0.99]

            for threshold in thresholds:
                filtered = [f for f in all_faces if f.confidence >= threshold]
                print(f"Faces with confidence >= {threshold}: {len(filtered)}")

        else:
            print("Image not found")

    except Exception as e:
        print(f"Note: {e}")


def example_5_face_size_filtering() -> None:
    """Example 5: Filter detections by face size."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Face Size Filtering")
    print("="*60)

    try:
        detector = RetinaFaceDetector()

        image_path = 'test_image.jpg'
        image, faces = detector.detect(image_path)

        if image is not None and faces:
            # Calculate face sizes
            large_faces = []
            small_faces = []
            medium_faces = []

            min_large = 200
            min_medium = 100

            for face in faces:
                x1, y1, x2, y2 = face.bbox
                width = x2 - x1
                height = y2 - y1
                size = max(width, height)

                if size >= min_large:
                    large_faces.append((face, size))
                elif size >= min_medium:
                    medium_faces.append((face, size))
                else:
                    small_faces.append((face, size))

            print(f"\nFace Size Distribution:")
            print(f"  Large (>={min_large}px):   {len(large_faces)} faces")
            print(f"  Medium (>={min_medium}px): {len(medium_faces)} faces")
            print(f"  Small (<{min_medium}px):   {len(small_faces)} faces")

            if large_faces:
                avg_size = sum(s for _, s in large_faces) / len(large_faces)
                print(f"\n  Average large face size: {avg_size:.0f}px")

        else:
            print("No faces detected or image not found")

    except Exception as e:
        print(f"Note: {e}")


def example_6_statistics() -> None:
    """Example 6: Generate and display detection statistics."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Detection Statistics")
    print("="*60)

    try:
        detector = RetinaFaceDetector()

        image_path = 'test_image.jpg'
        image, faces = detector.detect(image_path)

        if image is not None:
            # Get summary
            summary = detector.get_detection_summary(faces)

            print(f"\nDetection Summary:")
            print(f"  Total Faces:        {summary['total_faces']}")
            print(f"  Avg Confidence:     {summary['avg_confidence']:.4f}")
            print(f"  Min Confidence:     {summary['min_confidence']:.4f}")
            print(f"  Max Confidence:     {summary['max_confidence']:.4f}")

            if summary['bbox_sizes']:
                sizes = summary['bbox_sizes']
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]

                print(f"\nFace Size Statistics:")
                print(f"  Avg Width:  {np.mean(widths):.0f}px")
                print(f"  Avg Height: {np.mean(heights):.0f}px")
                print(f"  Min Width:  {np.min(widths):.0f}px")
                print(f"  Max Width:  {np.max(widths):.0f}px")

        else:
            print("Image not found")

    except Exception as e:
        print(f"Note: {e}")


def example_7_multiple_images() -> None:
    """Example 7: Process multiple images in a directory."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Batch Processing Multiple Images")
    print("="*60)

    try:
        detector = RetinaFaceDetector(output_dir='processed/detected_batch')

        # Create test directory structure if needed
        image_dir = Path('test_images')

        if image_dir.exists():
            image_files = list(image_dir.glob('*.jpg')) + \
                          list(image_dir.glob('*.png'))

            if image_files:
                total_faces = 0

                for image_path in image_files:
                    image, faces = detector.detect(image_path)

                    if image is not None:
                        total_faces += len(faces)
                        print(f"{image_path.name}: {len(faces)} faces")

                print(f"\n✓ Processed {len(image_files)} images")
                print(f"✓ Total faces detected: {total_faces}")

            else:
                print("No images found in test_images/")
        else:
            print("test_images/ directory not found")
            print("Create test_images/ with .jpg/.png files to test")

    except Exception as e:
        print(f"Note: {e}")


def example_8_extract_face_regions() -> None:
    """Example 8: Extract and save individual face regions."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Extract Individual Face Regions")
    print("="*60)

    try:
        detector = RetinaFaceDetector()

        image_path = 'test_image.jpg'
        image, faces = detector.detect(image_path)

        if image is not None and faces:
            # Create output directory
            output_dir = Path('extracted_faces')
            output_dir.mkdir(exist_ok=True)

            print(f"Extracting {len(faces)} face region(s)...\n")

            for i, face in enumerate(faces):
                # Get bounding box
                x1, y1, x2, y2 = face.bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Extract face region
                face_region = image[y1:y2, x1:x2]

                # Add padding (optional)
                padding = 10
                y1_pad = max(0, y1 - padding)
                x1_pad = max(0, x1 - padding)
                y2_pad = min(image.shape[0], y2 + padding)
                x2_pad = min(image.shape[1], x2 + padding)

                face_region_padded = image[y1_pad:y2_pad, x1_pad:x2_pad]

                # Save
                filename = (f"face_{i:03d}_"
                           f"conf_{face.confidence:.3f}.jpg")
                output_path = output_dir / filename

                cv2.imwrite(str(output_path), face_region_padded)

                print(f"  Saved: {filename}")
                print(f"    Size: {face_region_padded.shape[1]}x"
                      f"{face_region_padded.shape[0]}px")

            print(f"\n✓ Extracted {len(faces)} faces to: {output_dir}/")

        else:
            print("No faces detected or image not found")

    except Exception as e:
        print(f"Note: {e}")


def example_9_data_augmentation_prep() -> None:
    """Example 9: Prepare data for model training."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Prepare Data for Training")
    print("="*60)

    try:
        detector = RetinaFaceDetector()

        image_path = 'test_image.jpg'
        image, faces = detector.detect(image_path)

        if image is not None and faces:
            print(f"\nPreparing {len(faces)} face(s) for training:\n")

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Extract face
                face_region = image[y1:y2, x1:x2]

                # Preprocessing
                # 1. Resize to standard size
                target_size = (112, 112)
                face_resized = cv2.resize(face_region, target_size)

                # 2. Normalize to [0, 1]
                face_normalized = face_resized.astype(np.float32) / 255.0

                # 3. Convert BGR to RGB if needed
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                print(f"Face {i+1}:")
                print(f"  Original shape: {face_region.shape}")
                print(f"  Resized shape: {face_resized.shape}")
                print(f"  Normalized dtype: {face_normalized.dtype}")
                print(f"  Normalized range: [{face_normalized.min():.3f}, "
                      f"{face_normalized.max():.3f}]")
                print()

        else:
            print("No faces detected or image not found")

    except Exception as e:
        print(f"Note: {e}")


def example_10_integration() -> None:
    """Example 10: Integration with dataset loader."""
    print("\n" + "="*60)
    print("EXAMPLE 10: Integration with Dataset Loader")
    print("="*60)

    try:
        from dataset_loader import DatasetLoader

        print("Loading dataset...")
        loader = DatasetLoader()
        metadata_list = loader.load()

        if not metadata_list:
            print("No images in dataset")
            return

        detector = RetinaFaceDetector(output_dir='processed/detected_dataset')

        print(f"\nProcessing {len(metadata_list)} images...\n")

        total_faces = 0
        images_processed = 0

        # Process first 5 images as demo
        for metadata in metadata_list[:5]:
            try:
                image, faces = detector.detect(str(metadata.image_path))

                if image is not None:
                    total_faces += len(faces)
                    images_processed += 1

                    print(f"{metadata.filename}: {len(faces)} faces detected")

            except Exception as e:
                print(f"{metadata.filename}: Error - {e}")

        print(f"\n✓ Processed {images_processed} images")
        print(f"✓ Total faces detected: {total_faces}")

    except ImportError:
        print("dataset_loader module not available")
        print("Make sure dataset_loader.py is in the same directory")


def main() -> None:
    """Run all examples."""
    # Configure logging - use WARNING to reduce noise
    logging.basicConfig(level=logging.WARNING)

    print("\n" + "="*60)
    print("RetinaFaceDetector - Usage Examples")
    print("="*60)

    try:
        example_1_basic_detection()
        example_2_visualization()
        example_3_landmark_extraction()
        example_4_confidence_filtering()
        example_5_face_size_filtering()
        example_6_statistics()
        example_7_multiple_images()
        example_8_extract_face_regions()
        example_9_data_augmentation_prep()
        example_10_integration()

        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60 + "\n")

        print("Note: Some examples require test images.")
        print("To run with your images:")
        print("  1. Place test_image.jpg in current directory")
        print("  2. Or place images in test_images/ directory")
        print("  3. Run this script again")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
