from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PersonEntry:
    """Represents a person record stored in the database."""

    person_id: str
    name: str
    enrolled_at: str
    metadata: Dict[str, Any]
    active: bool = True


class DatabaseManager:
    """Database manager for person registration and metadata storage."""

    def __init__(
        self,
        database_dir: Path,
        persons_file: str = "persons.json",
        metadata_file: str = "metadata.json",
    ) -> None:
        self.database_dir = database_dir
        self.persons_path = database_dir / persons_file
        self.metadata_path = database_dir / metadata_file
        self.database_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Initializing DatabaseManager at %s", self.database_dir)
        self._ensure_files_exist()

    def _ensure_files_exist(self) -> None:
        if not self.persons_path.exists():
            self._write_json(self.persons_path, [])
            logger.debug("Created persons database file: %s", self.persons_path)

        if not self.metadata_path.exists():
            self._write_json(self.metadata_path, {})
            logger.debug("Created metadata database file: %s", self.metadata_path)

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _load_persons(self) -> List[Dict[str, Any]]:
        return self._read_json(self.persons_path)

    def _load_metadata(self) -> Dict[str, Any]:
        return self._read_json(self.metadata_path)

    def _save_persons(self, persons: List[Dict[str, Any]]) -> None:
        self._write_json(self.persons_path, persons)
        logger.debug("Saved %d person records", len(persons))

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        self._write_json(self.metadata_path, metadata)
        logger.debug("Saved metadata with %d keys", len(metadata))

    def add_person(self, person: PersonEntry, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new person to the database."""
        persons = self._load_persons()
        if any(record["person_id"] == person.person_id for record in persons):
            raise ValueError(f"Person ID '{person.person_id}' already exists.")

        persons.append(asdict(person))
        self._save_persons(persons)
        logger.info("Added person %s to database", person.person_id)

        if metadata is not None:
            stored_metadata = self._load_metadata()
            stored_metadata[person.person_id] = metadata
            self._save_metadata(stored_metadata)
            logger.info("Added metadata for person %s", person.person_id)

    def delete_person(self, person_id: str) -> bool:
        """Delete a person and their metadata from the database."""
        persons = self._load_persons()
        new_persons = [record for record in persons if record["person_id"] != person_id]

        if len(new_persons) == len(persons):
            logger.warning("Attempted to delete missing person_id: %s", person_id)
            return False

        self._save_persons(new_persons)
        metadata = self._load_metadata()

        if person_id in metadata:
            del metadata[person_id]
            self._save_metadata(metadata)
            logger.info("Deleted metadata for person %s", person_id)

        logger.info("Deleted person %s from database", person_id)
        return True

    def update_person(self, person_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields of a person entry."""
        persons = self._load_persons()
        updated = False

        for record in persons:
            if record["person_id"] == person_id:
                record.update(updates)
                updated = True
                break

        if not updated:
            logger.warning("Attempted update for missing person_id: %s", person_id)
            return False

        self._save_persons(persons)
        logger.info("Updated person %s with %s", person_id, updates)
        return True

    def search_person(self, query: str) -> List[Dict[str, Any]]:
        """Search persons by ID or name."""
        query_lower = query.strip().lower()
        persons = self._load_persons()

        return [
            record
            for record in persons
            if query_lower in record.get("person_id", "").lower()
            or query_lower in record.get("name", "").lower()
        ]

    def list_persons(self) -> List[Dict[str, Any]]:
        """Return all registered persons."""
        return self._load_persons()

    def get_person_metadata(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a registered person."""
        return self._load_metadata().get(person_id)

    def export_database(self, export_path: Path) -> Path:
        """Export the current database files to a destination folder."""
        export_path.mkdir(parents=True, exist_ok=True)
        destination_persons = export_path / self.persons_path.name
        destination_metadata = export_path / self.metadata_path.name

        self._write_json(destination_persons, self._load_persons())
        self._write_json(destination_metadata, self._load_metadata())

        logger.info("Exported database to %s", export_path)
        return export_path

    def backup_database(self, backup_dir: Path) -> Path:
        """Create a timestamped backup of the database files."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        self.export_database(backup_path)
        logger.info("Created backup at %s", backup_path)
        return backup_path

    def statistics(self) -> Dict[str, int]:
        """Return simple statistics for the database."""
        persons = self._load_persons()
        metadata = self._load_metadata()

        return {
            "total_persons": len(persons),
            "metadata_entries": len(metadata),
        }
