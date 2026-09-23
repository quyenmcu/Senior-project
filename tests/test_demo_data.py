import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from apps.dashboard.demo_data import kaggle_demo_summary


def test_kaggle_demo_summary_preserves_dataset_meaning():
    summary = kaggle_demo_summary()
    assert summary["raw_total"] == 3644
    assert summary["feature_total"] == 1102
    assert {row["label"] for row in summary["motion_counts"]} == {
        "Normal", "Aggressive", "Slow",
    }


@pytest.mark.django_db
def test_dashboard_labels_kaggle_data_as_demo(client):
    user = User.objects.create_user("demo-driver")
    client.force_login(user)
    response = client.get(reverse("dashboard"))
    assert response.status_code == 200
    assert b"Kaggle smartphone telemetry preview" in response.content
    assert b"excluded from your personal risk" in response.content
