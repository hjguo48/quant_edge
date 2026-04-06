from sqlalchemy import text
from src.data.db.session import get_engine

def check_date_range():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT ticker, MIN(trade_date), MAX(trade_date) FROM stock_prices GROUP BY ticker"))
        rows = result.all()
        for row in rows:
            print(f"{row[0]}: {row[1]} to {row[2]}")

if __name__ == "__main__":
    check_date_range()
