import base64
import json
from unittest.mock import patch

import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from apps.dashboard.views import _fatigue_daily_trend
from apps.driving.models import DrivingSession, FatigueEvent


@pytest.mark.django_db
def test_monitor_requires_login_and_sets_csrf_cookie(client):
    assert client.get(reverse("fatigue-monitor")).status_code == 302
    user = User.objects.create_user("camera-driver")
    client.force_login(user)
    response = client.get(reverse("fatigue-monitor"))
    assert response.status_code == 200
    assert "csrftoken" in response.cookies
    assert b"Calibrate Left" in response.content
    assert b"Calibrate Right" in response.content
    assert b"Head pose" in response.content


@pytest.mark.django_db
def test_monitor_renders_one_working_detection_interface(client):
    user = User.objects.create_user("single-monitor-driver")
    client.force_login(user)

    content = client.get(reverse("fatigue-monitor")).content.decode()

    assert content.count('id="startButton"') == 1
    assert content.count('id="camera"') == 1
    assert content.count('id="sessionState"') == 1
    assert content.count("let stream=") == 1


@pytest.mark.django_db
def test_admin_overview_is_staff_only(client):
    regular = User.objects.create_user("regular")
    client.force_login(regular)
    assert client.get(reverse("admin-overview")).status_code == 302
    staff = User.objects.create_user("staff", is_staff=True)
    client.force_login(staff)
    response = client.get(reverse("admin-overview"))
    assert response.status_code == 200
    assert b"Registered users" in response.content


@pytest.mark.django_db
def test_daily_fatigue_trend_groups_events_and_fills_empty_days():
    user = User.objects.create_user("trend-driver")
    now = timezone.now()
    session = DrivingSession.objects.create(user=user, started_at=now)
    FatigueEvent.objects.create(
        user=user,
        session=session,
        recorded_at=now,
        status=FatigueEvent.Status.DROWSY,
        confidence=0.9,
    )
    trend = _fatigue_daily_trend(user.fatigue_events.all(), days=3)
    assert len(trend["labels"]) == 3
    drowsy = next(item for item in trend["series"] if item["name"] == "Drowsy")
    assert drowsy["values"] == [0, 0, 1]


@pytest.mark.django_db
@patch("apps.driving.fatigue_detector.analyze")
def test_analyzed_status_transition_is_saved(mock_analyze, client, settings, tmp_path):
    import cv2
    import numpy as np

    settings.MEDIA_ROOT = tmp_path
    user = User.objects.create_user("detected-driver")
    client.force_login(user)
    started = client.post(
        reverse("api-start-session"), data="{}", content_type="application/json"
    ).json()
    mock_analyze.return_value = {
        "detected": True,
        "calibrating": False,
        "status": "DROWSY",
        "confidence": 0.91,
        "ear": 0.18,
        "mar": 0.31,
        "average_ear": 0.29,
        "changed": True,
    }
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
    response = client.post(
        reverse("api-analyze-fatigue", args=[started["session_id"]]),
        data=json.dumps({"image": payload}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert user.fatigue_events.filter(status="DROWSY").count() == 1
    snapshot = user.fatigue_snapshots.get()
    assert snapshot.image.name.endswith(".jpg")

    owner_response = client.get(reverse("fatigue-snapshot", args=[snapshot.id]))
    assert owner_response.status_code == 200
    other = User.objects.create_user("other-driver")
    client.force_login(other)
    assert client.get(reverse("fatigue-snapshot", args=[snapshot.id])).status_code == 404
