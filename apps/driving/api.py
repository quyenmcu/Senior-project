import base64
import json
from uuid import uuid4

from django.core.files.base import ContentFile
from django.http import FileResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_POST

from .models import DrivingSession, FatigueEvent, FatigueSnapshot, TelemetryEvent


def _payload(request):
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc


def _timestamp(value):
    if not value:
        return timezone.now()
    from django.utils.dateparse import parse_datetime

    parsed = parse_datetime(value)
    if parsed is None:
        raise ValueError("recorded_at must be an ISO-8601 datetime.")
    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed)
    return parsed


def _owned_session(request, session_id):
    return get_object_or_404(DrivingSession, id=session_id, user=request.user)


@require_POST
def start_session(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    try:
        data = _payload(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    session = DrivingSession.objects.create(
        user=request.user,
        started_at=timezone.now(),
        source_device_id=str(data.get("source_device_id", ""))[:128],
    )
    return JsonResponse({"session_id": str(session.id), "started_at": session.started_at.isoformat()}, status=201)


@require_POST
def finish_session(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    session = _owned_session(request, session_id)
    session.ended_at = timezone.now()
    session.save(update_fields=["ended_at"])
    from .fatigue_detector import clear_session

    clear_session(session_id)
    return JsonResponse({"ended_at": session.ended_at.isoformat()})


@require_POST
def ingest_fatigue(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    session = _owned_session(request, session_id)
    try:
        data = _payload(request)
        status = str(data["status"]).upper()
        if status not in FatigueEvent.Status.values:
            raise ValueError(f"status must be one of {list(FatigueEvent.Status.values)}")
        event = FatigueEvent.objects.create(
            user=request.user,
            session=session,
            recorded_at=_timestamp(data.get("recorded_at")),
            status=status,
            confidence=data.get("confidence"),
            ear=data.get("ear"),
            mar=data.get("mar"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({"event_id": event.id}, status=201)


@require_POST
def analyze_fatigue_frame(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    session = _owned_session(request, session_id)
    try:
        import cv2
        import numpy as np

        from .fatigue_detector import analyze

        data = _payload(request)
        encoded = data["image"].split(",", 1)[-1]
        image_bytes = base64.b64decode(encoded)
        frame = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise ValueError("The submitted camera frame is invalid.")
        result = analyze(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), session.id)
        if result.get("changed"):
            event = FatigueEvent.objects.create(
                user=request.user,
                session=session,
                recorded_at=timezone.now(),
                status=result["status"],
                confidence=result["confidence"],
                ear=result["ear"],
                mar=result["mar"],
            )
            if result["status"] in {"DROWSY", "SLEEPING", "YAWNING"}:
                snapshot = FatigueSnapshot(
                    user=request.user,
                    session=session,
                    event=event,
                )
                snapshot.image.save(
                    f"{result['status'].lower()}-{uuid4().hex}.jpg",
                    ContentFile(image_bytes),
                    save=True,
                )
        return JsonResponse(result)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)


def fatigue_snapshot(request, snapshot_id):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    snapshots = FatigueSnapshot.objects.select_related("event")
    if not request.user.is_staff:
        snapshots = snapshots.filter(user=request.user)
    snapshot = get_object_or_404(snapshots, id=snapshot_id)
    return FileResponse(snapshot.image.open("rb"), content_type="image/jpeg")


@require_POST
def ingest_telemetry(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required."}, status=401)
    session = _owned_session(request, session_id)
    try:
        data = _payload(request)
        motion_class = str(data.get("motion_class", "NORMAL")).upper()
        if motion_class not in TelemetryEvent.MotionClass.values:
            raise ValueError(f"motion_class must be one of {list(TelemetryEvent.MotionClass.values)}")
        event = TelemetryEvent(
            user=request.user,
            session=session,
            recorded_at=_timestamp(data.get("recorded_at")),
            acc_x=float(data["acc_x"]),
            acc_y=float(data["acc_y"]),
            acc_z=float(data["acc_z"]),
            gyro_x=float(data["gyro_x"]),
            gyro_y=float(data["gyro_y"]),
            gyro_z=float(data["gyro_z"]),
            speed_kmh=float(data["speed_kmh"]) if data.get("speed_kmh") is not None else None,
            motion_class=motion_class,
        )
        event.full_clean()
        event.save()
    except (KeyError, TypeError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({"event_id": event.id}, status=201)
