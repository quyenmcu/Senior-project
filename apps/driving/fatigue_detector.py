from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
CALIBRATION_FRAMES = 50
MAR_THRESHOLD = 0.60


@dataclass
class SessionState:
    calibration: list[float] = field(default_factory=list)
    ear_history: deque = field(default_factory=lambda: deque(maxlen=5))
    mar_history: deque = field(default_factory=lambda: deque(maxlen=5))
    last_status: str | None = None


_states: dict[str, SessionState] = {}
_face_mesh = None


def _mesh():
    global _face_mesh
    if _face_mesh is None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                'MediaPipe is not installed. Run: python -m pip install -e ".[dev,cv]"'
            ) from exc
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _face_mesh


def _distance(a, b):
    return hypot(a.x - b.x, a.y - b.y)


def _ear(points, indices):
    eye = [points[index] for index in indices]
    horizontal = _distance(eye[0], eye[3])
    if horizontal <= 1e-8:
        return 0.0
    return (_distance(eye[1], eye[5]) + _distance(eye[2], eye[4])) / (2 * horizontal)


def _mar(points):
    vertical = _distance(points[MOUTH[0]], points[MOUTH[1]])
    horizontal = _distance(points[MOUTH[2]], points[MOUTH[3]])
    return vertical / horizontal if horizontal > 1e-8 else 0.0


def analyze(rgb_frame, session_id):
    result = _mesh().process(rgb_frame)
    if not result.multi_face_landmarks:
        return {"detected": False, "message": "No face detected"}

    points = result.multi_face_landmarks[0].landmark
    current_ear = (_ear(points, LEFT_EYE) + _ear(points, RIGHT_EYE)) / 2
    current_mar = _mar(points)
    state = _states.setdefault(str(session_id), SessionState())

    if len(state.calibration) < CALIBRATION_FRAMES:
        state.calibration.append(current_ear)
        return {
            "detected": True,
            "calibrating": True,
            "progress": len(state.calibration),
            "required": CALIBRATION_FRAMES,
            "ear": round(current_ear, 3),
            "mar": round(current_mar, 3),
            **_landmarks(points),
        }

    average_ear = sum(state.calibration) / len(state.calibration)
    state.ear_history.append(current_ear)
    state.mar_history.append(current_mar)
    ear = sum(state.ear_history) / len(state.ear_history)
    mar = sum(state.mar_history) / len(state.mar_history)

    if ear < average_ear * 0.75:
        status = "SLEEPING"
        confidence = min(1.0, (average_ear * 0.75 - ear) / max(average_ear * 0.75, 1e-8) + 0.70)
    elif ear < average_ear * 0.90:
        status = "DROWSY"
        confidence = min(1.0, (average_ear * 0.90 - ear) / max(average_ear * 0.15, 1e-8))
    elif mar > MAR_THRESHOLD:
        status = "YAWNING"
        confidence = min(1.0, 0.70 + (mar - MAR_THRESHOLD))
    else:
        status = "ACTIVE"
        confidence = min(1.0, ear / max(average_ear, 1e-8))

    changed = status != state.last_status
    state.last_status = status
    return {
        "detected": True,
        "calibrating": False,
        "status": status,
        "confidence": round(confidence, 3),
        "ear": round(ear, 3),
        "mar": round(mar, 3),
        "average_ear": round(average_ear, 3),
        "changed": changed,
        **_landmarks(points),
    }


def _landmarks(points):
    used = sorted(set(LEFT_EYE + RIGHT_EYE + MOUTH))
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return {
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "landmarks": {
            str(index): [round(points[index].x, 5), round(points[index].y, 5)]
            for index in used
        },
        "left_eye": LEFT_EYE,
        "right_eye": RIGHT_EYE,
        "mouth": MOUTH,
    }


def clear_session(session_id):
    _states.pop(str(session_id), None)
