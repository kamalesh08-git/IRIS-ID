from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ui.deployment import DeploymentManager


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Periocular Recognition Application Manager"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["start", "stop", "status", "export", "backup"],
        default="status",
        help="Command to execute.",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("output/export"),
        help="Destination folder for database export.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_arguments()

    project_root = Path(__file__).resolve().parent
    manager = DeploymentManager(project_root=project_root)

    if args.command == "start":
        manager.start()
        logging.info("Recognition service is running. Press Enter to stop.")
        input("Press Enter to shut down the application...\n")
        manager.stop()
    elif args.command == "stop":
        manager.stop()
    elif args.command == "status":
        context = manager.get_dashboard_context()
        logging.info("Application status: %s", context["status"])
        logging.info("Total users: %s", context["status_details"]["Total Users"])
        logging.info("History events: %s", context["status_details"]["History Events"])
    elif args.command == "export":
        exported_path = manager.export_database(args.export_path)
        logging.info("Database exported to %s", exported_path)
    elif args.command == "backup":
        backup_path = manager.backup_database()
        logging.info("Database backup created at %s", backup_path)


if __name__ == "__main__":
    main()
