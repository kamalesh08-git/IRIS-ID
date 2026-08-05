from __future__ import annotations

from ui.config_manager import ConfigManager
from ui.database_manager import DatabaseManager, PersonEntry
from ui.dashboard import Dashboard, PersonRecord, RecognitionHistoryEntry
from ui.deployment import DeploymentManager

__all__ = [
    "ConfigManager",
    "DatabaseManager",
    "PersonEntry",
    "Dashboard",
    "PersonRecord",
    "RecognitionHistoryEntry",
    "DeploymentManager",
]
