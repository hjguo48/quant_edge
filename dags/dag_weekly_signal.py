from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum


def check_data_freshness() -> dict[str, str]:
    return {"status": "ok", "step": "check_data_freshness"}


def compute_features() -> dict[str, str]:
    return {"status": "ok", "step": "compute_features"}


def model_inference() -> dict[str, str]:
    return {"status": "ok", "step": "model_inference"}


def signal_risk_check() -> dict[str, str]:
    return {"status": "ok", "step": "signal_risk_check"}


with DAG(
    dag_id="weekly_signal_pipeline",
    description="Friday signal generation and Layer 2 risk checks.",
    schedule="30 16 * * 5",
    start_date=pendulum.datetime(2026, 1, 2, tz="America/New_York"),
    catchup=False,
    tags=["quantedge", "signals", "weekly"],
    default_args={"owner": "quantedge"},
) as dag:
    check_data_freshness_task = PythonOperator(
        task_id="check_data_freshness",
        python_callable=check_data_freshness,
    )
    compute_features_task = PythonOperator(
        task_id="compute_features",
        python_callable=compute_features,
    )
    model_inference_task = PythonOperator(
        task_id="model_inference",
        python_callable=model_inference,
    )
    signal_risk_check_task = PythonOperator(
        task_id="signal_risk_check",
        python_callable=signal_risk_check,
    )

    check_data_freshness_task >> compute_features_task >> model_inference_task >> signal_risk_check_task
