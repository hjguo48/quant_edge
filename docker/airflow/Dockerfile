FROM apache/airflow:2.9.3-python3.11

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/quantedge-airflow-requirements.txt

RUN python -m pip install --no-cache-dir --root-user-action ignore \
    -r /tmp/quantedge-airflow-requirements.txt

USER airflow
