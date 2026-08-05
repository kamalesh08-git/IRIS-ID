from __future__ import annotations

import json
import logging
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Iterable, List, Optional

from ui.config_manager import ConfigManager
from ui.database_manager import DatabaseManager, PersonEntry
from ui.dashboard import Dashboard, PersonRecord, RecognitionHistoryEntry

logger = logging.getLogger(__name__)


class DeploymentManager:
    """Application deployment manager for startup, shutdown, and runtime orchestration."""

    def __init__(
        self,
        project_root: Path,
        database_dir: Optional[Path] = None,
        config_file: Optional[Path] = None,
    ) -> None:
        self.project_root = project_root
        self.database_dir = database_dir or project_root / "database"
        self.config_file = config_file or project_root / "config.json"
        logger.debug(
            "DeploymentManager created with project_root=%s, database_dir=%s, config_file=%s",
            project_root,
            self.database_dir,
            self.config_file,
        )

        self.config_manager = ConfigManager(self.project_root)
        self.database_manager = DatabaseManager(self.database_dir)
        self.dashboard = Dashboard()
        self._running = False

    def initialize(self) -> None:
        """Initialize the application and prepare all modules."""
        self.config_manager.ensure_output_directories()
        self.config_manager.validate_paths()
        self._register_signal_handlers()
        logger.info("Application initialized successfully")

    def _register_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        logger.debug("Registered shutdown signal handlers")

    def _shutdown_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        logger.info("Received shutdown signal %s", signum)
        self.shutdown()
        sys.exit(0)

    def start(self) -> None:
        """Start the recognition service and UI orchestration."""
        if self._running:
            logger.warning("DeploymentManager start called while already running")
            return

        self.initialize()
        self._running = True
        logger.info("Application startup complete")

    def stop(self) -> None:
        """Stop the recognition service gracefully."""
        if not self._running:
            logger.warning("DeploymentManager stop called when not running")
            return

        self._running = False
        logger.info("Application stopped")

    def shutdown(self) -> None:
        """Shutdown the application and release resources."""
        self.stop()
        logger.info("Application shutdown completed")

    def get_dashboard_context(self) -> Dict[str, Any]:
        """Return current dashboard state data."""
        persons = self.database_manager.list_persons()
        history = self._load_history()
        live_results = self._load_live_results()
        status_details = {
            "Total Users": str(len(persons)),
            "History Events": str(len(history)),
            "Confidence": f"{self.config_manager.get_threshold('recognition_confidence'):.0%}",
        }

        return {
            "persons": [
                PersonRecord(
                    person_id=record["person_id"],
                    name=record["name"],
                    enrollment_date=record["enrolled_at"],
                    metadata=record.get("metadata", {}),
                    status="Active" if record.get("active", True) else "Inactive",
                )
                for record in persons
            ],
            "history": [
                RecognitionHistoryEntry(
                    timestamp=entry.get("timestamp", "-"),
                    person_id=entry.get("person_id", "-"),
                    name=entry.get("name", "Unknown"),
                    confidence=entry.get("confidence", 0.0),
                    status=entry.get("status", "Pending"),
                    notes=entry.get("notes"),
                )
                for entry in history
            ],
            "live_results": live_results,
            "status": "Running" if self._running else "Stopped",
            "status_details": status_details,
        }

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load recognition history from the database or persistent store."""
        history_path = self.project_root / "database" / "history.json"
        if not history_path.exists():
            logger.debug("Recognition history file not found: %s", history_path)
            return []

        try:
            with history_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as error:
            logger.error("Failed to load recognition history: %s", error)
            return []

    def _load_live_results(self) -> List[Dict[str, Any]]:
        """Simulate or read the current live recognition report stream."""
        live_path = self.project_root / "database" / "live_results.json"
        if not live_path.exists():
            logger.debug("Live results file not found: %s", live_path)
            return []

        try:
            with live_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as error:
            logger.error("Failed to load live recognition results: %s", error)
            return []

    def register_person(self, person_id: str, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new person in the system."""
        person = PersonEntry(
            person_id=person_id,
            name=name,
            enrolled_at=str(Path().resolve()),
            metadata=metadata or {},
            active=True,
        )
        self.database_manager.add_person(person, metadata=metadata)
        logger.info("Registered new person %s", person_id)

    def delete_person(self, person_id: str) -> bool:
        """Delete an existing person from the system."""
        result = self.database_manager.delete_person(person_id)
        logger.info("Deleted person %s: %s", person_id, result)
        return result

    def export_database(self, export_path: Path) -> Path:
        """Export the database to the configured export folder."""
        destination = self.config_manager.project_root / self.config_manager.get_setting("output_folders", "export")
        logger.info("Exporting database to %s", export_path)
        return self.database_manager.export_database(export_path)

    def backup_database(self) -> Path:
        """Backup the database to the configured backup folder."""
        backup_dir = self.project_root / "database" / "backups"
        logger.info("Backing up database to %s", backup_dir)
        return self.database_manager.backup_database(backup_dir)
