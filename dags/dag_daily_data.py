from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum


def fetch_prices() -> dict[str, str]:
    return {"status": "ok", "step": "fetch_prices"}


def check_quality() -> dict[str, str]:
    return {"status": "ok", "step": "check_quality"}


def store_to_db() -> dict[str, str]:
    return {"status": "ok", "step": "store_to_db"}


def update_features_cache() -> dict[str, str]:
    return {"status": "ok", "step": "update_features_cache"}


with DAG(
    dag_id="daily_data_pipeline",
    description="Daily market data ingestion and cache refresh.",
    schedule="0 2 * * 1-5",
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    tags=["quantedge", "data", "daily"],
    default_args={"owner": "quantedge"},
) as dag:
    fetch_prices_task = PythonOperator(task_id="fetch_prices", python_callable=fetch_prices)
    check_quality_task = PythonOperator(task_id="check_quality", python_callable=check_quality)
    store_to_db_task = PythonOperator(task_id="store_to_db", python_callable=store_to_db)
    update_features_cache_task = PythonOperator(
        task_id="update_features_cache",
        python_callable=update_features_cache,
    )

    fetch_prices_task >> check_quality_task >> store_to_db_task >> update_features_cache_task
