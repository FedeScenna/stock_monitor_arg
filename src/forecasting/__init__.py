"""
Multi-model price forecasting for CEDEARs.

Provides a uniform `Forecaster` interface so the Kronos foundation model and the
Nixtla deep time-series models (N-HiTS, PatchTST, TFT) can be benchmarked head to
head and blended into an ensemble.

    from src.forecasting import ForecastResult, NeuralForecaster, KronosForecaster, ensemble
"""
from src.forecasting.base import Forecaster, ForecastResult, next_trading_days
from src.forecasting.ensemble import ensemble, weights_from_backtest

# NeuralForecaster (neuralforecast) and KronosForecaster (torch + Kronos) pull in
# heavy deps, so import them directly from their submodules when needed:
#     from src.forecasting.neural import NeuralForecaster
#     from src.forecasting.kronos_model import KronosForecaster

__all__ = [
    "Forecaster",
    "ForecastResult",
    "next_trading_days",
    "ensemble",
    "weights_from_backtest",
]
