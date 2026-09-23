import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils.crypto import get_random_string


@pytest.mark.django_db
def test_registration_creates_and_logs_in_user(client):
    response = client.post(reverse("register"), {
        "username": "driver1", "first_name": "Test", "last_name": "Driver",
        "email": "driver@example.com", "password1": "Safe-test-password-917!",
        "password2": "Safe-test-password-917!",
    })
    generated_password = get_random_string(32)
    response = client.post(
        reverse("register"),
        {
            "username": "driver1",
            "first_name": "Test",
            "last_name": "Driver",
            "email": "driver@example.com",
            "password1": generated_password,
            "password2": generated_password,
        },
    )
    assert response.status_code == 302
    assert User.objects.filter(username="driver1").exists()


@pytest.mark.django_db
def test_dashboard_requires_login(client):
    response = client.get(reverse("dashboard"))
    assert response.status_code == 302
    assert reverse("login") in response.url
