"""Command-line entrypoint for managing DocIntel database migrations."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from psycopg import connect

from ..config import VectorDatabaseSettings, get_vector_db_settings
from .migrations import apply_migrations, get_pending_migrations


def _configure_settings(dsn_override: str | None) -> VectorDatabaseSettings:
    settings = get_vector_db_settings()
    updates = {}
    if dsn_override:
        updates["dsn"] = dsn_override
        updates["enabled"] = True
    if updates:
        settings = settings.model_copy(update=updates)
    if not settings.enabled:
        raise SystemExit(
            "Vector database is disabled. Set DOCINTEL_VECTOR_DB_ENABLED=1 or provide --dsn to run migrations."
        )
    if not settings.dsn:
        raise SystemExit(
            "No PostgreSQL connection string provided. Set DOCINTEL_VECTOR_DB_DSN or pass --dsn."  # noqa: TRY003
        )
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply DocIntel pgvector migrations")
    parser.add_argument(
        "--dsn",
        help="Optional PostgreSQL connection string overriding DOCINTEL_VECTOR_DB_DSN.",
        default=None,
    )
    parser.add_argument(
        "--target-version",
        type=int,
        default=None,
        help="Apply migrations up to and including the specified version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list pending migrations without applying them.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = _configure_settings(args.dsn)

    dsn = str(settings.dsn)

    with connect(dsn) as conn:
        with conn.cursor() as cursor:
            timeout_ms = int(settings.statement_timeout_seconds * 1000)
            cursor.execute(f"SET statement_timeout = {timeout_ms}")
        pending = get_pending_migrations(conn, settings)
        if args.dry_run:
            if not pending:
                logging.info("db | migrations | database is up to date")
            else:
                for migration in pending:
                    logging.info(
                        "db | migrations | pending | version=%s | name=%s", migration.version, migration.name
                    )
            return 0

        applied = apply_migrations(conn, settings, target_version=args.target_version)
        if applied:
            for migration in applied:
                logging.info(
                    "db | migrations | applied | version=%s | name=%s", migration.version, migration.name
                )
        else:
            logging.info("db | migrations | null | database already current")

    return 0


if __name__ == "__main__":
    sys.exit(main())
