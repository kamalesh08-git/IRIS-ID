from __future__ import annotations

import logging
from pathlib import Path

from ui.deployment import DeploymentManager


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parent.parent
    manager = DeploymentManager(project_root=project_root)
    manager.start()
    logging.info("DeploymentManager is running. Press Enter to stop.")
    input("Press Enter to shut down the application...\n")
    manager.stop()


if __name__ == "__main__":
    main()
