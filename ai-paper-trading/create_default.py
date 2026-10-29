
from app.api.services.strategy_profile_service import StrategyProfileService


service = StrategyProfileService()

service.create(
    name="Default",
    strategy="ema_rsi",
    parameters={
        "ema_fast": 20,
        "ema_slow": 50,
        "rsi_period": 14,
        "rsi_buy": 30,
        "rsi_sell": 70,
        "stop_loss": 150,
        "take_profit": 300,
        "risk_percent": 1.0,
    },
)