import uuid

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models


class DriverProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="driver_profile")
    annual_distance_km = models.PositiveIntegerField(default=10000)
    vehicle_label = models.CharField(max_length=120, blank=True)


class DrivingSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="driving_sessions")
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)
    source_device_id = models.CharField(max_length=128, blank=True)

    class Meta:
        ordering = ["-started_at"]


class FatigueEvent(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE", "Active"
        DROWSY = "DROWSY", "Drowsy"
        YAWNING = "YAWNING", "Yawning"
        SLEEPING = "SLEEPING", "Sleeping"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="fatigue_events")
    session = models.ForeignKey(DrivingSession, on_delete=models.CASCADE, related_name="fatigue_events")
    recorded_at = models.DateTimeField(db_index=True)
    status = models.CharField(max_length=12, choices=Status.choices)
    confidence = models.FloatField(null=True, blank=True)
    ear = models.FloatField(null=True, blank=True)
    mar = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-recorded_at"]
        indexes = [models.Index(fields=["user", "recorded_at"])]


def fatigue_snapshot_path(instance, filename):
    return f"fatigue/{instance.user_id}/{instance.session_id}/{filename}"


class FatigueSnapshot(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="fatigue_snapshots"
    )
    session = models.ForeignKey(
        DrivingSession, on_delete=models.CASCADE, related_name="fatigue_snapshots"
    )
    event = models.OneToOneField(
        FatigueEvent, on_delete=models.CASCADE, related_name="snapshot"
    )
    image = models.ImageField(upload_to=fatigue_snapshot_path)
    captured_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-captured_at"]


class TelemetryEvent(models.Model):
    class MotionClass(models.TextChoices):
        SLOW = "SLOW", "Slow"
        NORMAL = "NORMAL", "Normal"
        AGGRESSIVE = "AGGRESSIVE", "Aggressive"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="telemetry_events")
    session = models.ForeignKey(DrivingSession, on_delete=models.CASCADE, related_name="telemetry_events")
    recorded_at = models.DateTimeField(db_index=True)
    acc_x = models.FloatField()
    acc_y = models.FloatField()
    acc_z = models.FloatField()
    gyro_x = models.FloatField()
    gyro_y = models.FloatField()
    gyro_z = models.FloatField()
    speed_kmh = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0)])
    motion_class = models.CharField(max_length=12, choices=MotionClass.choices, default=MotionClass.NORMAL)

    class Meta:
        ordering = ["-recorded_at"]
        indexes = [models.Index(fields=["user", "recorded_at"])]


class PremiumEstimate(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="premium_estimates")
    calculated_at = models.DateTimeField(auto_now_add=True)
    period_start = models.DateField()
    period_end = models.DateField()
    fatigue_risk = models.FloatField()
    driving_risk = models.FloatField()
    total_risk = models.FloatField()
    annual_distance_km = models.PositiveIntegerField()
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    model_version = models.CharField(max_length=40, default="formula-v1")

    class Meta:
        ordering = ["-period_start"]
