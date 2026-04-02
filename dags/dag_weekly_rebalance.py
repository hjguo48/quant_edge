from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum


def load_signals() -> dict[str, str]:
    return {"status": "ok", "step": "load_signals"}


def portfolio_optimize() -> dict[str, str]:
    return {"status": "ok", "step": "portfolio_optimize"}


def portfolio_risk_check() -> dict[str, str]:
    return {"status": "ok", "step": "portfolio_risk_check"}


def generate_orders() -> dict[str, str]:
    return {"status": "ok", "step": "generate_orders"}


def audit_log() -> dict[str, str]:
    return {"status": "ok", "step": "audit_log"}


with DAG(
    dag_id="weekly_rebalance_pipeline",
    description="Friday rebalance orchestration with portfolio risk controls.",
    schedule="0 17 * * 5",
    start_date=pendulum.datetime(2026, 1, 2, tz="America/New_York"),
    catchup=False,
    tags=["quantedge", "rebalance", "weekly"],
    default_args={"owner": "quantedge"},
) as dag:
    load_signals_task = PythonOperator(task_id="load_signals", python_callable=load_signals)
    portfolio_optimize_task = PythonOperator(
        task_id="portfolio_optimize",
        python_callable=portfolio_optimize,
    )
    portfolio_risk_check_task = PythonOperator(
        task_id="portfolio_risk_check",
        python_callable=portfolio_risk_check,
    )
    generate_orders_task = PythonOperator(
        task_id="generate_orders",
        python_callable=generate_orders,
    )
    audit_log_task = PythonOperator(task_id="audit_log", python_callable=audit_log)

    load_signals_task >> portfolio_optimize_task >> portfolio_risk_check_task >> generate_orders_task >> audit_log_task
