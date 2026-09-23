from datetime import timedelta

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie

from apps.driving.models import FatigueEvent
from apps.driving.services import premium_forecast, premium_time_series, summarize_user

from .demo_data import kaggle_demo_summary


def home(request):
    return render(request, "dashboard/home.html")


@login_required
def dashboard(request):
    summary = summarize_user(request.user)
    fatigue = request.user.fatigue_events.select_related("session")[:10]
    telemetry = request.user.telemetry_events.select_related("session")[:10]
    behavior = (
        request.user.telemetry_events.values("motion_class")
        .order_by()
        .annotate(total=Count("id"))
    )
    weekly_forecasts = premium_forecast(request.user, "weekly", 8)
    monthly_forecasts = premium_forecast(request.user, "monthly", 6)
    premium_series = premium_time_series(request.user)
    fatigue_counts = {
        row["status"]: row["total"]
        for row in request.user.fatigue_events.values("status").annotate(total=Count("id"))
    }
    fatigue_chart = _fatigue_chart(fatigue_counts)
    fatigue_trend = _fatigue_daily_trend(request.user.fatigue_events.all())
    snapshots = request.user.fatigue_snapshots.select_related("event")[:6]
    kaggle_demo = kaggle_demo_summary() if not telemetry else None
    return render(request, "dashboard/index.html", {
        "summary": summary,
        "fatigue_events": fatigue,
        "telemetry_events": telemetry,
        "behavior_summary": behavior,
        "weekly_forecasts": weekly_forecasts,
        "monthly_forecasts": monthly_forecasts,
        "premium_series": premium_series,
        "fatigue_chart": fatigue_chart,
        "fatigue_trend": fatigue_trend,
        "fatigue_total": sum(fatigue_counts.values()),
        "fatigue_alerts": sum(
            fatigue_counts.get(status, 0) for status in ("DROWSY", "SLEEPING", "YAWNING")
        ),
        "snapshots": snapshots,
        "kaggle_demo": kaggle_demo,
    })


@login_required
@ensure_csrf_cookie
def fatigue_monitor(request):
    return render(request, "dashboard/fatigue_monitor.html")


@staff_member_required
def admin_overview(request):
    users = User.objects.order_by("-date_joined")
    rows = []
    overall_counts = {
        row["status"]: row["total"]
        for row in FatigueEvent.objects.values("status").annotate(total=Count("id"))
    }
    fatigue_trend = _fatigue_daily_trend(FatigueEvent.objects.all())
    for user in users:
        summary = summarize_user(user)
        user_counts = {
            row["status"]: row["total"]
            for row in user.fatigue_events.values("status").annotate(total=Count("id"))
        }
        rows.append({
            "user": user,
            "fatigue_count": user.fatigue_events.count(),
            "telemetry_count": user.telemetry_events.count(),
            "session_count": user.driving_sessions.count(),
            "summary": summary,
            "counts": user_counts,
        })
    return render(request, "dashboard/admin_overview.html", {
        "rows": rows,
        "user_count": users.count(),
        "active_count": users.filter(is_active=True).count(),
        "fatigue_chart": _fatigue_chart(overall_counts),
        "fatigue_trend": fatigue_trend,
        "fatigue_total": sum(overall_counts.values()),
    })


def _fatigue_chart(counts):
    colors = {
        "ACTIVE": "#22c55e",
        "DROWSY": "#f5c518",
        "SLEEPING": "#ef4444",
        "YAWNING": "#f97316",
    }
    maximum = max(counts.values(), default=0)
    return [
        {
            "status": status.title(),
            "count": counts.get(status, 0),
            "width": (counts.get(status, 0) / maximum * 100) if maximum else 0,
            "color": colors[status],
        }
        for status in ("ACTIVE", "DROWSY", "SLEEPING", "YAWNING")
    ]


def _fatigue_daily_trend(events, days=14):
    """Return a complete daily series, including zero-event days."""
    end = timezone.localdate()
    start = end - timedelta(days=days - 1)
    rows = (
        events.filter(recorded_at__date__gte=start, recorded_at__date__lte=end)
        .annotate(day=TruncDate("recorded_at"))
        .values("day", "status")
        .annotate(total=Count("id"))
        .order_by("day")
    )
    lookup = {(row["day"], row["status"]): row["total"] for row in rows}
    labels = [start + timedelta(days=offset) for offset in range(days)]
    statuses = ("ACTIVE", "DROWSY", "SLEEPING", "YAWNING")
    colors = {
        "ACTIVE": "#22c55e",
        "DROWSY": "#f59e0b",
        "SLEEPING": "#ef4444",
        "YAWNING": "#8b5cf6",
    }
    return {
        "labels": [day.isoformat() for day in labels],
        "series": [
            {
                "name": status.title(),
                "color": colors[status],
                "values": [lookup.get((day, status), 0) for day in labels],
            }
            for status in statuses
        ],
    }
