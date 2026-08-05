from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "model_paths": {
        "face_detector": "models/retinaface.pth",
        "recognition_model": "models/arcface.pth",
    },
    "thresholds": {
        "recognition_confidence": 0.75,
        "quality_score": 0.5,
    },
    "output_folders": {
        "export": "output/export",
        "logs": "logs",
        "embeddings": "database/embeddings",
    },
    "environment": {
        "APP_ENV": "development",
        "LOG_LEVEL": "INFO",
    },
}


@dataclass
class ConfigManager:
    """Manage configuration files, paths, thresholds, and environment variables."""

    project_root: Path
    config_file: Path = field(init=False)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.config_file = self.project_root / "config.json"
        logger.debug("ConfigManager initialized with config_file=%s", self.config_file)
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file or initialize defaults."""
        if self.config_file.exists():
            try:
                with self.config_file.open("r", encoding="utf-8") as handle:
                    self.config = json.load(handle)
                logger.info("Loaded configuration from %s", self.config_file)
            except (OSError, ValueError) as error:
                logger.error("Failed to read config file: %s", error)
                self.config = DEFAULT_CONFIG.copy()
        else:
            self.config = DEFAULT_CONFIG.copy()
            self.save_config()
            logger.info("Initialized default configuration to %s", self.config_file)

    def save_config(self) -> None:
        """Save the current configuration dictionary to file."""
        try:
            with self.config_file.open("w", encoding="utf-8") as handle:
                json.dump(self.config, handle, indent=2, ensure_ascii=False)
            logger.info("Configuration saved to %s", self.config_file)
        except OSError as error:
            logger.error("Unable to save config file: %s", error)
            raise

    def get_setting(self, *keys: str, default: Optional[Any] = None) -> Any:
        """Retrieve a nested setting by keys."""
        current = self.config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                logger.debug("Setting %s not found, returning default", keys)
                return default
            current = current[key]
        return current

    def set_setting(self, value: Any, *keys: str) -> None:
        """Update a nested setting and persist the config file."""
        current: Dict[str, Any] = self.config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
        self.save_config()
        logger.debug("Set setting %s to %s", keys, value)

    def reset_defaults(self) -> None:
        """Reset configuration to defaults and save."""
        self.config = DEFAULT_CONFIG.copy()
        self.save_config()
        logger.info("Configuration reset to defaults")

    def ensure_output_directories(self) -> Dict[str, Path]:
        """Ensure configured output directories exist."""
        output_dirs: Dict[str, Path] = {}
        for name, relative_path in self.get_setting("output_folders", default={}).items():
            destination = self.project_root / Path(relative_path)
            destination.mkdir(parents=True, exist_ok=True)
            output_dirs[name] = destination
            logger.debug("Ensured output directory %s exists at %s", name, destination)
        return output_dirs

    def validate_paths(self) -> Dict[str, bool]:
        """Validate configured model and output paths."""
        results: Dict[str, bool] = {}

        for name, relative_path in self.get_setting("model_paths", default={}).items():
            path = self.project_root / Path(relative_path)
            results[f"model_paths.{name}"] = path.exists()
            logger.debug("Validated model path %s: %s", path, results[f"model_paths.{name}"])

        for name, relative_path in self.get_setting("output_folders", default={}).items():
            path = self.project_root / Path(relative_path)
            results[f"output_folders.{name}"] = path.exists()
            logger.debug("Validated output folder %s: %s", path, results[f"output_folders.{name}"])

        return results

    def get_threshold(self, threshold_name: str) -> float:
        """Return a numeric threshold from configuration."""
        value = self.get_setting("thresholds", threshold_name)
        if value is None:
            raise KeyError(f"Threshold '{threshold_name}' not configured.")
        return float(value)

    def get_env(self, name: str, default: Optional[str] = None) -> str:
        """Read an environment variable, falling back to config defaults."""
        value = os.getenv(name)
        if value is not None:
            logger.debug("Environment override %s=%s", name, value)
            return value
        return str(self.get_setting("environment", name, default=default))

    def set_env(self, name: str, value: str) -> None:
        """Set an environment variable and update config defaults."""
        os.environ[name] = value
        if "environment" not in self.config:
            self.config["environment"] = {}
        self.config["environment"][name] = value
        self.save_config()
        logger.debug("Set environment %s=%s", name, value)

    def refresh(self) -> None:
        """Reload configuration from disk into memory."""
        self.load_config()
        logger.info("Configuration reloaded from disk")
