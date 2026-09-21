# IrisMind architecture

The Django application is the source of truth for identity, driving sessions, fatigue events,
smartphone telemetry, risk summaries, premium estimates, and dashboard views.

1. A registered user starts a `DrivingSession`.
2. The fatigue client writes `FatigueEvent` records for that authenticated session.
3. The smartphone client writes `TelemetryEvent` records for the same session.
4. The risk service aggregates records owned by the signed-in user.
5. The dashboard displays current status, driving behavior, premium, and projections.

Initial session-authenticated ingestion contracts:

- `POST /api/sessions/start/`
- `POST /api/sessions/<session_id>/fatigue/`
- `POST /api/sessions/<session_id>/telemetry/`

The next implementation stage will expose authenticated ingestion endpoints for the fatigue
client and smartphone application. Raw images and video must remain private media artifacts;
ordinary dashboard and pricing queries should use event metadata only.
