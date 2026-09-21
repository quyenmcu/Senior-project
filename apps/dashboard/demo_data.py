import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

from django.conf import settings

MOTION_LABELS = ("NORMAL", "AGGRESSIVE", "SLOW")
EVENT_LABELS = {
    "1": "Sudden acceleration",
    "2": "Sudden right turn",
    "3": "Sudden left turn",
    "4": "Sudden braking",
}


@lru_cache(maxsize=1)
def kaggle_demo_summary():
    """Summarize bundled public demonstration data without creating user records."""
    data_dir = Path(settings.BASE_DIR) / "demo_data"
    raw_path = data_dir / "train_motion_data.csv"
    feature_path = data_dir / "features_14.csv"
    if not raw_path.exists() or not feature_path.exists():
        return None

    motion_counts = Counter()
    samples = []
    with raw_path.open(newline="", encoding="utf-8-sig") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=1):
            motion_counts[row["Class"]] += 1
            if len(samples) < 5:
                samples.append({
                    "number": index,
                    "motion_class": row["Class"].title(),
                    "acc": f'{float(row["AccX"]):.3f}, {float(row["AccY"]):.3f}, {float(row["AccZ"]):.3f}',
                    "gyro": f'{float(row["GyroX"]):.3f}, {float(row["GyroY"]):.3f}, {float(row["GyroZ"]):.3f}',
                })

    event_counts = Counter()
    with feature_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            event_counts[row["Target"]] += 1

    return {
        "raw_total": sum(motion_counts.values()),
        "feature_total": sum(event_counts.values()),
        "motion_counts": [
            {"label": label.title(), "count": motion_counts[label]}
            for label in MOTION_LABELS
        ],
        "event_counts": [
            {"label": EVENT_LABELS[target], "count": event_counts[target]}
            for target in EVENT_LABELS
        ],
        "samples": samples,
    }
