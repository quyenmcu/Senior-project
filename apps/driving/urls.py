from django.urls import path

from . import api

urlpatterns = [
    path("sessions/start/", api.start_session, name="api-start-session"),
    path("sessions/<uuid:session_id>/finish/", api.finish_session, name="api-finish-session"),
    path("sessions/<uuid:session_id>/fatigue/", api.ingest_fatigue, name="api-ingest-fatigue"),
    path("sessions/<uuid:session_id>/fatigue/analyze/", api.analyze_fatigue_frame, name="api-analyze-fatigue"),
    path("sessions/<uuid:session_id>/telemetry/", api.ingest_telemetry, name="api-ingest-telemetry"),
    path("snapshots/<int:snapshot_id>/", api.fatigue_snapshot, name="fatigue-snapshot"),
]
