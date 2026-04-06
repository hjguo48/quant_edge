from src.api.routers.backtest import router as backtest_router
from src.api.routers.market import router as market_router
from src.api.routers.portfolio import router as portfolio_router
from src.api.routers.predictions import router as predictions_router
from src.api.routers.stocks import router as stocks_router

ROUTERS = (
    market_router,
    stocks_router,
    predictions_router,
    portfolio_router,
    backtest_router,
)

__all__ = ["ROUTERS"]
