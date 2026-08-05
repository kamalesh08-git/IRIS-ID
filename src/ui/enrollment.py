from __future__ import annotations

import json
import logging
import mimetypes
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ui.config_manager import ConfigManager
from ui.database_manager import DatabaseManager, PersonEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrollmentResult:
    """Represents the result of a completed enrollment operation."""

    person_id: str
    name: str
    enrolled_at: str
    image_count: int
    metadata: Dict[str, Any]
    api_response: Dict[str, Any]


class EnrollmentManager:
    """Manage person enrollment, image validation, and metadata storage."""

    def __init__(
        self,
        project_root: Path,
        database_dir: Optional[Path] = None,
        config_manager: Optional[ConfigManager] = None,
    ) -> None:
        self.project_root = project_root
        self.config_manager = config_manager or ConfigManager(project_root)
        self.database_manager = DatabaseManager(database_dir or project_root / "database")
        self.enrollment_dir = project_root / "database" / "enrollment"
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_image_types = {"image/jpeg", "image/png", "image/bmp"}

        logger.debug(
            "EnrollmentManager initialized: project_root=%s, enrollment_dir=%s",
            project_root,
            self.enrollment_dir,
        )

    def choose_images(self, file_paths: Iterable[Path]) -> List[Path]:
        """Return a list of candidate image paths from user selection."""
        selected = [path for path in file_paths if path.exists() and path.is_file()]
        logger.debug("choose_images selected %d files", len(selected))
        return selected

    def validate_images(self, image_paths: Iterable[Path]) -> List[Path]:
        """Validate a set of image files before enrollment."""
        validated: List[Path] = []

        for image_path in image_paths:
            if not image_path.exists():
                message = f"Image file not found: {image_path}"
                logger.error(message)
                raise FileNotFoundError(message)

            if not image_path.is_file():
                message = f"Path is not a file: {image_path}"
                logger.error(message)
                raise ValueError(message)

            mime_type, _ = mimetypes.guess_type(str(image_path))
            if mime_type not in self.allowed_image_types:
                message = f"Unsupported image format: {image_path}"
                logger.error(message)
                raise ValueError(message)

            validated.append(image_path)

        logger.info("Validated %d enrollment images", len(validated))
        return validated

    def call_member_two_api(self, image_paths: Iterable[Path]) -> Dict[str, Any]:
        """Call Member 2's recognition service API (stubbed placeholder)."""
        processed = [str(path.name) for path in image_paths]
        response = {
            "status": "success",
            "processed_images": len(processed),
            "files": processed,
            "message": "Images prepared for recognition pipeline.",
        }
        logger.debug("call_member_two_api response=%s", response)
        return response

    def _copy_images_for_person(self, person_id: str, image_paths: Iterable[Path]) -> Path:
        """Copy validated images into a dedicated enrollment directory."""
        destination_dir = self.enrollment_dir / person_id
        destination_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            destination_path = destination_dir / image_path.name
            if image_path.resolve() != destination_path.resolve():
                shutil.copy2(image_path, destination_path)
                logger.debug("Copied %s to %s", image_path, destination_path)

        return destination_dir

    def register_person(
        self,
        person_id: str,
        name: str,
        image_paths: Iterable[Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnrollmentResult:
        """Register a new person and store enrollment metadata."""
        images = self.validate_images(image_paths)
        api_response = self.call_member_two_api(images)
        copied_dir = self._copy_images_for_person(person_id, images)

        person_metadata = {
            "name": name,
            "image_count": len(images),
            "enrollment_directory": str(copied_dir),
            **(metadata or {}),
        }

        entry = PersonEntry(
            person_id=person_id,
            name=name,
            enrolled_at=datetime.utcnow().isoformat() + "Z",
            metadata=person_metadata,
        )

        self.database_manager.add_person(entry, metadata=person_metadata)
        logger.info("Registered person %s with %d images", person_id, len(images))

        return EnrollmentResult(
            person_id=person_id,
            name=name,
            enrolled_at=entry.enrolled_at,
            image_count=len(images),
            metadata=person_metadata,
            api_response=api_response,
        )

    def update_person_metadata(self, person_id: str, updates: Dict[str, Any]) -> bool:
        """Update enrollment metadata for an existing person."""
        result = self.database_manager.update_person(person_id, updates)
        logger.info("Update person metadata for %s: %s", person_id, result)
        return result

    def preview_enrollment(self, person_id: str) -> Dict[str, Any]:
        """Return stored enrollment metadata for preview."""
        metadata = self.database_manager.get_person_metadata(person_id)
        if metadata is None:
            logger.warning("No metadata found for person_id=%s", person_id)
            return {}
        logger.debug("Preview enrollment metadata for %s", person_id)
        return metadata

    def export_enrollment_metadata(self, export_path: Path) -> Path:
        """Export enrollment metadata as a JSON file."""
        export_path.mkdir(parents=True, exist_ok=True)
        metadata = {
            person["person_id"]: self.database_manager.get_person_metadata(person["person_id"])
            for person in self.database_manager.list_persons()
        }
        destination = export_path / "enrollment_metadata.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        logger.info("Exported enrollment metadata to %s", destination)
        return destination
