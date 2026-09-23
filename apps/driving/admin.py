from django.contrib import admin

from .models import (
    DriverProfile,
    DrivingSession,
    FatigueEvent,
    FatigueSnapshot,
    PremiumEstimate,
    TelemetryEvent,
)

admin.site.register(
    [
        DriverProfile,
        DrivingSession,
        FatigueEvent,
        FatigueSnapshot,
        TelemetryEvent,
        PremiumEstimate,
    ]
)
