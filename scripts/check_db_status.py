from sqlalchemy import text
from src.data.db.session import get_engine

def check_db():
    engine = get_engine()
    tables = [
        "stocks",
        "stock_prices",
        "fundamentals_pit",
        "universe_membership",
        "corporate_actions",
        "feature_store",
        "model_registry",
        "predictions",
        "portfolios",
        "backtest_results",
        "audit_log"
    ]
    
    with engine.connect() as conn:
        print(f"{'Table':<25} | {'Count':<10}")
        print("-" * 40)
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"{table:<25} | {count:<10}")
            except Exception as e:
                print(f"{table:<25} | Error: {e}")

if __name__ == "__main__":
    check_db()
