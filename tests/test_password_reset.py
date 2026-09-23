import pytest
from django.contrib.auth.models import User
from django.core import mail
from django.urls import reverse
from django.utils.crypto import get_random_string


@pytest.mark.django_db
def test_password_reset_sends_email_for_registered_address(client, settings):
    settings.EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
    User.objects.create_user(
        "driver-reset", email="driver@example.com", password=get_random_string(32)
    )
    response = client.post(reverse("password_reset"), {"email": "driver@example.com"})
    assert response.status_code == 302
    assert response.url == reverse("password_reset_done")
    assert len(mail.outbox) == 1
    assert "reset" in mail.outbox[0].subject.lower()
    assert "/accounts/reset/" in mail.outbox[0].body
