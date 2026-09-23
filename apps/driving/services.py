from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from django.conf import settings
from django.db.models import Count
from django.utils import timezone

from .models import DriverProfile

FATIGUE_RISK = {"ACTIVE": 0, "DROWSY": 1, "YAWNING": 2, "SLEEPING": 3}
DRIVING_RISK = {"SLOW": 1, "NORMAL": 1, "AGGRESSIVE": 3}


@dataclass(frozen=True)
class RiskSummary:
    fatigue_risk: float
    driving_risk: float
    total_risk: float
    premium: Decimal


def summarize_user(user) -> RiskSummary:
    fatigue_rows = user.fatigue_events.values("status").annotate(total=Count("id"))
    telemetry_rows = user.telemetry_events.values("motion_class").annotate(total=Count("id"))

    def weighted_average(rows, mapping, default):
        total = sum(row["total"] for row in rows)
        if not total:
            return default
        return sum(mapping[row[next(k for k in row if k != "total")]] * row["total"] for row in rows) / total

    fatigue = weighted_average(list(fatigue_rows), FATIGUE_RISK, 0.0)
    driving = weighted_average(list(telemetry_rows), DRIVING_RISK, 0.0)
    total = (fatigue + driving) / 2
    profile, _ = DriverProfile.objects.get_or_create(
        user=user, defaults={"annual_distance_km": settings.IRISMIND_DEFAULT_ANNUAL_KM}
    )
    amount = Decimal(str(settings.IRISMIND_BASE_PREMIUM + total * settings.IRISMIND_RISK_RATE * profile.annual_distance_km)).quantize(Decimal("0.01"))
    return RiskSummary(fatigue, driving, total, amount)


def premium_forecast(user, interval="monthly", periods=6):
    """Project the annual premium estimate at weekly or monthly intervals."""
    if interval not in {"weekly", "monthly"}:
        raise ValueError("interval must be weekly or monthly")
    history = list(
        user.premium_estimates.order_by("period_start").values(
            "period_start", "amount"
        )
    )
    current = summarize_user(user).premium
    start_amount = history[-1]["amount"] if history else current
    daily_trend = Decimal(0)
    if len(history) >= 2:
        elapsed = (history[-1]["period_start"] - history[0]["period_start"]).days
        if elapsed > 0:
            daily_trend = (history[-1]["amount"] - history[0]["amount"]) / Decimal(elapsed)

    today = timezone.localdate()
    step_days = 7 if interval == "weekly" else 30
    floor = Decimal(str(settings.IRISMIND_BASE_PREMIUM))
    points = []
    for index in range(1, periods + 1):
        days = step_days * index
        amount = max(floor, start_amount + daily_trend * Decimal(days))
        points.append({
            "label": f"{'Week' if interval == 'weekly' else 'Month'} {index}",
            "date": today + timedelta(days=days),
            "amount": amount.quantize(Decimal("0.01")),
        })
    return points


def premium_time_series(user, future_periods=6):
    history = list(
        user.premium_estimates.order_by("period_start").values(
            "period_start", "amount"
        )
    )[-6:]
    forecast = premium_forecast(user, "monthly", future_periods)
    return {
        "history": [
            {"date": row["period_start"].isoformat(), "amount": float(row["amount"])}
            for row in history
        ],
        "forecast": [
            {"date": row["date"].isoformat(), "amount": float(row["amount"])}
            for row in forecast
        ],
        "has_trend": len(history) >= 2,
    }


def future_premiums(user, months=6):
    """Backward-compatible list used by older callers."""
    return [row["amount"] for row in premium_forecast(user, "monthly", months)]
