from sqlalchemy import text
from src.data.db.session import get_engine

def check_tickers():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT ticker, COUNT(*) FROM stock_prices GROUP BY ticker"))
        rows = result.all()
        for row in rows:
            print(f"{row[0]}: {row[1]}")

if __name__ == "__main__":
    check_tickers()
