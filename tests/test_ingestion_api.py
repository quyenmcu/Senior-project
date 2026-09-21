import json

import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from apps.driving.models import DrivingSession, FatigueEvent, TelemetryEvent


@pytest.mark.django_db
def test_authenticated_clients_can_create_a_shared_session_and_events(client):
    user = User.objects.create_user("driver", password="test-password")
    client.force_login(user)
    response = client.post(
        reverse("api-start-session"),
        data=json.dumps({"source_device_id": "phone-1"}),
        content_type="application/json",
    )
    assert response.status_code == 201
    session_id = response.json()["session_id"]

    fatigue = client.post(
        reverse("api-ingest-fatigue", args=[session_id]),
        data=json.dumps({"status": "DROWSY", "confidence": 0.88, "ear": 0.19, "mar": 0.32}),
        content_type="application/json",
    )
    telemetry = client.post(
        reverse("api-ingest-telemetry", args=[session_id]),
        data=json.dumps({
            "acc_x": 0.1, "acc_y": 0.2, "acc_z": 9.7,
            "gyro_x": 0.01, "gyro_y": 0.02, "gyro_z": 0.03,
            "speed_kmh": 42, "motion_class": "NORMAL",
        }),
        content_type="application/json",
    )
    assert fatigue.status_code == telemetry.status_code == 201
    assert FatigueEvent.objects.filter(user=user, session_id=session_id).count() == 1
    assert TelemetryEvent.objects.filter(user=user, session_id=session_id).count() == 1


@pytest.mark.django_db
def test_user_cannot_write_to_another_users_session(client):
    owner = User.objects.create_user("owner")
    attacker = User.objects.create_user("attacker")
    session = DrivingSession.objects.create(user=owner, started_at=timezone.now())
    client.force_login(attacker)
    response = client.post(
        reverse("api-ingest-fatigue", args=[session.id]),
        data=json.dumps({"status": "SLEEPING"}),
        content_type="application/json",
    )
    assert response.status_code == 404
