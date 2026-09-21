import pytest
from django.contrib.auth.models import User
from django.urls import reverse


@pytest.mark.django_db
def test_registration_creates_and_logs_in_user(client):
    response = client.post(reverse("register"), {
        "username": "driver1", "first_name": "Test", "last_name": "Driver",
        "email": "driver@example.com", "password1": "Safe-test-password-917!",
        "password2": "Safe-test-password-917!",
    })
    assert response.status_code == 302
    assert User.objects.filter(username="driver1").exists()


@pytest.mark.django_db
def test_dashboard_requires_login(client):
    response = client.get(reverse("dashboard"))
    assert response.status_code == 302
    assert reverse("login") in response.url
