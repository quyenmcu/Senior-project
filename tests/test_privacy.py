import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from apps.driving.models import DrivingSession, FatigueEvent


@pytest.mark.django_db
def test_dashboard_does_not_show_another_users_events(client):
    owner = User.objects.create_user("owner", password="test-password")
    intruder = User.objects.create_user("intruder", password="test-password")
    session = DrivingSession.objects.create(user=owner, started_at=timezone.now())
    FatigueEvent.objects.create(user=owner, session=session, recorded_at=timezone.now(), status="SLEEPING")
    client.force_login(intruder)
    response = client.get(reverse("dashboard"))
    assert response.status_code == 200
    assert b"0 total events" in response.content
    assert b"No fatigue events yet" in response.content
