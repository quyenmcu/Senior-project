from datetime import date
from decimal import Decimal

import pytest
from django.contrib.auth.models import User
from django.utils import timezone

from apps.driving.models import (
    DrivingSession,
    FatigueEvent,
    PremiumEstimate,
    TelemetryEvent,
)
from apps.driving.services import premium_forecast, premium_time_series, summarize_user


@pytest.mark.django_db
def test_empty_profile_starts_at_base_premium_without_assumed_risk():
    user = User.objects.create_user("new-driver")
    result = summarize_user(user)
    assert result.fatigue_risk == 0
    assert result.driving_risk == 0
    assert result.total_risk == 0
    assert result.premium == Decimal("300.00")


@pytest.mark.django_db
def test_summary_uses_only_owned_events():
    user = User.objects.create_user("driver")
    other = User.objects.create_user("other")
    session = DrivingSession.objects.create(user=user, started_at=timezone.now())
    other_session = DrivingSession.objects.create(user=other, started_at=timezone.now())
    FatigueEvent.objects.create(user=user, session=session, recorded_at=timezone.now(), status="DROWSY")
    FatigueEvent.objects.create(user=other, session=other_session, recorded_at=timezone.now(), status="SLEEPING")
    TelemetryEvent.objects.create(user=user, session=session, recorded_at=timezone.now(), acc_x=0, acc_y=0, acc_z=1, gyro_x=0, gyro_y=0, gyro_z=0, motion_class="NORMAL")
    result = summarize_user(user)
    assert result.fatigue_risk == 1
    assert result.driving_risk == 1
    assert result.total_risk == 1
    assert result.premium == Decimal("800.00")


@pytest.mark.django_db
def test_weekly_and_monthly_forecasts_use_elapsed_time_trend():
    user = User.objects.create_user("forecast-driver")
    common = {
        "user": user,
        "period_end": date(2026, 1, 31),
        "fatigue_risk": 0,
        "driving_risk": 0,
        "total_risk": 0,
        "annual_distance_km": 10000,
    }
    PremiumEstimate.objects.create(
        **common, period_start=date(2026, 1, 1), amount=Decimal("600.00")
    )
    PremiumEstimate.objects.create(
        **common, period_start=date(2026, 1, 31), amount=Decimal("630.00")
    )
    weekly = premium_forecast(user, "weekly", 2)
    monthly = premium_forecast(user, "monthly", 2)
    assert [point["amount"] for point in weekly] == [Decimal("637.00"), Decimal("644.00")]
    assert [point["amount"] for point in monthly] == [Decimal("660.00"), Decimal("690.00")]
    assert premium_time_series(user)["has_trend"] is True


@pytest.mark.django_db
def test_forecast_is_flat_without_enough_history():
    user = User.objects.create_user("new-forecast-driver")
    assert {point["amount"] for point in premium_forecast(user, "weekly", 4)} == {
        Decimal("300.00")
    }


@pytest.mark.django_db
def test_forecast_rejects_unknown_interval():
    user = User.objects.create_user("bad-interval-driver")
    with pytest.raises(ValueError):
        premium_forecast(user, "daily", 2)
