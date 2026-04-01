from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import time

from alembic import command
from alembic.config import Config
from loguru import logger
import sqlalchemy as sa

from _data_ops import CORE_TABLES, REPO_ROOT, check_http_endpoint, check_tcp_port, configure_logging
from src.config import settings
from src.data.db.session import get_engine
from src.data.sources.fred import MACRO_SERIES_TABLE


@dataclass(frozen=True)
class ServiceStatus:
    name: str
    container_running: bool | None
    endpoint_ready: bool


def main() -> int:
    configure_logging("init_db")

    service_statuses = _check_service_statuses()
    for status in service_statuses:
        container_state = (
            "running"
            if status.container_running is True
            else "not_running"
            if status.container_running is False
            else "unknown"
        )
        logger.info(
            "service {} container_status={} endpoint_ready={}",
            status.name,
            container_state,
            status.endpoint_ready,
        )

    if not next(status.endpoint_ready for status in service_statuses if status.name == "timescaledb"):
        logger.error("TimescaleDB endpoint is not reachable on {}:{}", settings.POSTGRES_HOST, settings.POSTGRES_PORT)
        return 1

    if not _wait_for_database():
        logger.error("TimescaleDB did not become ready in time")
        return 1

    _run_alembic_upgrade()
    _validate_core_tables()
    _verify_stock_prices_hypertable()
    _ensure_macro_series_table()
    _log_summary()

    logger.info("database initialization completed successfully")
    return 0


def _check_service_statuses() -> list[ServiceStatus]:
    running_services = _get_running_compose_services()
    return [
        ServiceStatus(
            name="timescaledb",
            container_running=_lookup_container_state(running_services, "timescaledb"),
            endpoint_ready=check_tcp_port(settings.POSTGRES_HOST, settings.POSTGRES_PORT),
        ),
        ServiceStatus(
            name="redis",
            container_running=_lookup_container_state(running_services, "redis"),
            endpoint_ready=check_tcp_port(settings.REDIS_HOST, settings.REDIS_PORT),
        ),
        ServiceStatus(
            name="mlflow",
            container_running=_lookup_container_state(running_services, "mlflow"),
            endpoint_ready=check_http_endpoint(settings.MLFLOW_TRACKING_URI),
        ),
    ]


def _get_running_compose_services() -> set[str] | None:
    compose_command = _get_compose_command()
    if compose_command is None:
        logger.warning("docker compose is unavailable; falling back to endpoint checks only")
        return None

    result = subprocess.run(
        [*compose_command, "ps", "--services", "--status", "running"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("docker compose ps failed: {}", result.stderr.strip() or result.stdout.strip())
        return None

    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }


def _get_compose_command() -> list[str] | None:
    docker_binary = shutil.which("docker")
    if docker_binary:
        result = subprocess.run(
            [docker_binary, "compose", "version"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return [docker_binary, "compose"]

    docker_compose_binary = shutil.which("docker-compose")
    if docker_compose_binary:
        return [docker_compose_binary]

    return None


def _lookup_container_state(running_services: set[str] | None, service_name: str) -> bool | None:
    if running_services is None:
        return None
    return service_name in running_services


def _wait_for_database(*, attempts: int = 30, sleep_seconds: float = 2.0) -> bool:
    engine = get_engine()
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as connection:
                connection.execute(sa.text("SELECT 1"))
            logger.info("TimescaleDB is ready after {} attempt(s)", attempt)
            return True
        except Exception as exc:
            logger.warning(
                "waiting for TimescaleDB readiness attempt {}/{}: {}",
                attempt,
                attempts,
                exc,
            )
            time.sleep(sleep_seconds)
    return False


def _run_alembic_upgrade() -> None:
    alembic_config = Config(str(Path(REPO_ROOT) / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(Path(REPO_ROOT) / "alembic"))
    logger.info("running alembic upgrade head")
    command.upgrade(alembic_config, "head")


def _validate_core_tables() -> None:
    inspector = sa.inspect(get_engine())
    existing_tables = set(inspector.get_table_names())
    missing = [table_name for table_name in CORE_TABLES if table_name not in existing_tables]
    if missing:
        raise RuntimeError(f"Core table validation failed; missing tables: {missing}")
    logger.info("validated {} core tables", len(CORE_TABLES))


def _verify_stock_prices_hypertable() -> None:
    engine = get_engine()
    statement = sa.text(
        """
        SELECT 1
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'public'
          AND hypertable_name = 'stock_prices'
        """,
    )
    with engine.connect() as connection:
        exists = connection.execute(statement).scalar_one_or_none()
    if exists != 1:
        raise RuntimeError("stock_prices is not registered as a TimescaleDB hypertable")
    logger.info("validated stock_prices hypertable registration")


def _ensure_macro_series_table() -> None:
    engine = get_engine()
    MACRO_SERIES_TABLE.metadata.create_all(engine, tables=[MACRO_SERIES_TABLE], checkfirst=True)
    inspector = sa.inspect(engine)
    if not inspector.has_table("macro_series_pit"):
        raise RuntimeError("macro_series_pit was not created successfully")
    logger.info("validated macro_series_pit table creation")


def _log_summary() -> None:
    inspector = sa.inspect(get_engine())
    logger.info(
        "initialization summary core_tables={} macro_series_pit_present={}",
        len(CORE_TABLES),
        inspector.has_table("macro_series_pit"),
    )


if __name__ == "__main__":
    sys.exit(main())
