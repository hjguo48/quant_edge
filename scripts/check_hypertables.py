from sqlalchemy import text
from src.data.db.session import get_engine

def check_hypertables():
    engine = get_engine()
    tables = ["stock_prices", "feature_store", "predictions"]
    
    with engine.connect() as conn:
        for table in tables:
            try:
                # TimescaleDB stores hypertable info in timescaledb_information.hypertables
                result = conn.execute(text(f"SELECT COUNT(*) FROM _timescaledb_catalog.hypertable WHERE table_name = '{table}'"))
                is_hyper = result.scalar() > 0
                print(f"{table}: {'Hypertable' if is_hyper else 'Regular Table'}")
            except Exception as e:
                print(f"{table}: Error: {e}")

if __name__ == "__main__":
    check_hypertables()
