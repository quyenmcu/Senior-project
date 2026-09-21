import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from apps.dashboard.demo_data import kaggle_demo_summary


@pytest.fixture
def demo_files(settings, tmp_path):
    data_dir = tmp_path / "demo_data"
    data_dir.mkdir()
    (data_dir / "train_motion_data.csv").write_text(
        "AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Class,Timestamp\n"
        "0,0,1,0,0,0,NORMAL,1\n"
        "1,0,1,0,1,0,AGGRESSIVE,2\n"
        "0,1,1,1,0,0,SLOW,3\n",
        encoding="utf-8",
    )
    (data_dir / "features_14.csv").write_text(
        "Target,AccMeanX\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n",
        encoding="utf-8",
    )
    settings.BASE_DIR = tmp_path
    kaggle_demo_summary.cache_clear()
    yield
    kaggle_demo_summary.cache_clear()


def test_kaggle_demo_summary_preserves_dataset_meaning(demo_files):
    summary = kaggle_demo_summary()
    assert summary["raw_total"] == 3
    assert summary["feature_total"] == 4
    assert {row["label"] for row in summary["motion_counts"]} == {
        "Normal", "Aggressive", "Slow",
    }


@pytest.mark.django_db
def test_dashboard_labels_kaggle_data_as_demo(client, demo_files):
    user = User.objects.create_user("demo-driver")
    client.force_login(user)
    response = client.get(reverse("dashboard"))
    assert response.status_code == 200
    assert b"Kaggle smartphone telemetry preview" in response.content
    assert b"excluded from your personal risk" in response.content


def test_demo_data_is_optional(settings, tmp_path):
    settings.BASE_DIR = tmp_path
    kaggle_demo_summary.cache_clear()
    assert kaggle_demo_summary() is None
