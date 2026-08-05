"""
Dataset Loader Module for Masked Face Recognition System.

This module provides utilities to recursively scan and load image datasets
from multiple sources (LFW, MFR2, RMFRD), validate image files, and extract
metadata for processing.

Production-quality implementation with robust error handling and logging.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

import cv2


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """
    Dataclass to store image metadata extracted from dataset.

    Attributes:
        image_path (Path): Absolute path to the image file.
        dataset_name (str): Name of the dataset (e.g., 'LFW', 'MFR2', 'RMFRD').
        filename (str): Base filename of the image.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
    """
    image_path: Path
    dataset_name: str
    filename: str
    width: int
    height: int

    def __repr__(self) -> str:
        """Return a readable representation of the metadata."""
        return (
            f"ImageMetadata("
            f"dataset={self.dataset_name}, "
            f"filename={self.filename}, "
            f"size={self.width}x{self.height})"
        )


class DatasetLoader:
    """
    Loads and validates image datasets from multiple sources.

    This class recursively scans specified dataset directories, validates image
    files, extracts metadata, and provides a summary of loaded datasets.

    Attributes:
        VALID_EXTENSIONS (Set[str]): Supported image file extensions.
        DATASET_PATHS (Dict[str, Path]): Mapping of dataset names to paths.
    """

    VALID_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png'}
    DATASET_PATHS: Dict[str, Path] = {
        'LFW': Path('datasets/lfw'),
        'MFR2': Path('datasets/mfr2'),
        'RMFRD': Path('datasets/rmfrd'),
    }

    def __init__(self) -> None:
        """Initialize the DatasetLoader."""
        self.metadata_list: List[ImageMetadata] = []
        self.summary: Dict[str, int] = defaultdict(int)
        logger.info("DatasetLoader initialized")

    def _is_valid_image(self, file_path: Path) -> bool:
        """
        Check if a file is a valid, readable image.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            bool: True if the file is a valid image, False otherwise.
        """
        # Check file extension
        if file_path.suffix.lower() not in self.VALID_EXTENSIONS:
            return False

        try:
            # Attempt to read the image with OpenCV
            image = cv2.imread(str(file_path))
            if image is None:
                logger.warning(f"Could not read image (corrupted or invalid): {file_path}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Error reading image {file_path}: {e}")
            return False

    def _extract_image_metadata(
        self, file_path: Path, dataset_name: str
    ) -> ImageMetadata | None:
        """
        Extract metadata from an image file.

        Args:
            file_path (Path): Path to the image file.
            dataset_name (str): Name of the dataset.

        Returns:
            ImageMetadata | None: Metadata object if successful, None otherwise.
        """
        try:
            # Read image to get dimensions
            image = cv2.imread(str(file_path))
            if image is None:
                return None

            height, width = image.shape[:2]
            filename = file_path.name

            metadata = ImageMetadata(
                image_path=file_path.resolve(),
                dataset_name=dataset_name,
                filename=filename,
                width=width,
                height=height,
            )
            return metadata
        except Exception as e:
            logger.warning(
                f"Failed to extract metadata from {file_path}: {e}"
            )
            return None

    def _load_dataset(self, dataset_name: str, dataset_path: Path) -> int:
        """
        Recursively load all valid images from a dataset directory.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'LFW').
            dataset_path (Path): Path to the dataset directory.

        Returns:
            int: Number of images successfully loaded from this dataset.
        """
        images_loaded = 0

        # Check if dataset path exists
        if not dataset_path.exists():
            logger.info(f"Dataset path does not exist: {dataset_path}")
            return images_loaded

        if not dataset_path.is_dir():
            logger.warning(f"Dataset path is not a directory: {dataset_path}")
            return images_loaded

        logger.info(f"Scanning dataset: {dataset_name} at {dataset_path}")

        # Recursively find all image files
        for file_path in dataset_path.rglob('*'):
            # Skip directories
            if file_path.is_dir():
                continue

            # Validate image file
            if not self._is_valid_image(file_path):
                continue

            # Extract metadata
            metadata = self._extract_image_metadata(file_path, dataset_name)
            if metadata is not None:
                self.metadata_list.append(metadata)
                self.summary[dataset_name] += 1
                images_loaded += 1
                logger.debug(f"Loaded: {metadata}")

        logger.info(
            f"Dataset '{dataset_name}' loading complete: "
            f"{images_loaded} images loaded"
        )
        return images_loaded

    def load(self) -> List[ImageMetadata]:
        """
        Load all datasets and return metadata list.

        This method:
        1. Scans all configured dataset directories
        2. Validates and loads image files
        3. Extracts metadata for each valid image
        4. Generates and displays a summary

        Returns:
            List[ImageMetadata]: List of metadata objects for all loaded images.
        """
        logger.info("Starting dataset loading process...")

        # Load each dataset
        for dataset_name, dataset_path in self.DATASET_PATHS.items():
            self._load_dataset(dataset_name, dataset_path)

        # Display summary
        self._display_summary()

        logger.info(
            f"Dataset loading complete. Total images: {len(self.metadata_list)}"
        )
        return self.metadata_list

    def _display_summary(self) -> None:
        """Display a formatted summary of loaded datasets."""
        print("\n" + "=" * 50)
        print("Dataset Summary")
        print("=" * 50)

        # Display per-dataset counts
        total_images = 0
        for dataset_name in self.DATASET_PATHS.keys():
            count = self.summary.get(dataset_name, 0)
            print(f"{dataset_name:10} : {count:5} images")
            total_images += count

        print("-" * 50)
        print(f"{'Total Images':10} : {total_images:5}")
        print("=" * 50 + "\n")

    def get_metadata_summary(self) -> Dict[str, int]:
        """
        Get a summary of image counts per dataset.

        Returns:
            Dict[str, int]: Dictionary mapping dataset names to image counts.
        """
        return dict(self.summary)

    def get_metadata_list(self) -> List[ImageMetadata]:
        """
        Get the complete list of loaded image metadata.

        Returns:
            List[ImageMetadata]: List of all loaded image metadata.
        """
        return self.metadata_list

    def filter_by_dataset(self, dataset_name: str) -> List[ImageMetadata]:
        """
        Filter metadata by dataset name.

        Args:
            dataset_name (str): Name of the dataset to filter by.

        Returns:
            List[ImageMetadata]: Metadata for images from the specified dataset.
        """
        return [
            metadata for metadata in self.metadata_list
            if metadata.dataset_name == dataset_name
        ]

    def filter_by_size_range(
        self, min_width: int = 0, min_height: int = 0,
        max_width: int = float('inf'), max_height: int = float('inf')
    ) -> List[ImageMetadata]:
        """
        Filter metadata by image size range.

        Args:
            min_width (int): Minimum image width in pixels.
            min_height (int): Minimum image height in pixels.
            max_width (int): Maximum image width in pixels.
            max_height (int): Maximum image height in pixels.

        Returns:
            List[ImageMetadata]: Filtered metadata based on size criteria.
        """
        return [
            metadata for metadata in self.metadata_list
            if (min_width <= metadata.width <= max_width and
                min_height <= metadata.height <= max_height)
        ]


def main() -> None:
    """
    Main entry point for the dataset loader.

    Demonstrates the DatasetLoader functionality by loading all configured
    datasets and displaying a summary.
    """
    logger.info("Starting Masked Face Recognition Dataset Loader")

    # Initialize and load datasets
    loader = DatasetLoader()
    metadata_list = loader.load()

    # Display statistics
    summary = loader.get_metadata_summary()
    logger.info(f"Loading complete. Summary: {summary}")

    # Example: Display first few images (if any loaded)
    if metadata_list:
        logger.info("\nFirst 5 loaded images:")
        for i, metadata in enumerate(metadata_list[:5], 1):
            logger.info(f"  {i}. {metadata}")
    else:
        logger.warning("No images were loaded. Check dataset paths.")


if __name__ == '__main__':
    main()
