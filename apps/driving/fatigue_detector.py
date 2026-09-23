from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import atan2, degrees, hypot

import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)
CALIBRATION_FRAMES = 50
SIDE_CALIBRATION_FRAMES = 30
MAR_THRESHOLD = 0.60


def normalize_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def classify_head_pose(pitch: float, roll: float) -> str:
    if roll < -20.0:
        return "LEFT"
    if roll > 20.0:
        return "RIGHT"
    if pitch > 12.0:
        return "FORWARD"
    if pitch < -10.0:
        return "BACKWARD"
    return "NORMAL"


@dataclass
class HeadPoseClassifier:
    current: str = "NORMAL"

    def update(self, pitch: float, roll: float) -> str:
        if self.current == "LEFT" and roll < -14.0:
            return self.current
        if self.current == "RIGHT" and roll > 14.0:
            return self.current
        if roll < -20.0:
            self.current = "LEFT"
            return self.current
        if roll > 20.0:
            self.current = "RIGHT"
            return self.current
        if self.current == "FORWARD" and pitch > 8.0:
            return self.current
        if self.current == "BACKWARD" and pitch < -7.0:
            return self.current
        self.current = classify_head_pose(pitch, roll)
        return self.current


@dataclass
class SessionState:
    left_calibration: list[float] = field(default_factory=list)
    right_calibration: list[float] = field(default_factory=list)
    pitch_calibration: list[float] = field(default_factory=list)
    yaw_calibration: list[float] = field(default_factory=list)
    roll_calibration: list[float] = field(default_factory=list)
    left_ear_history: deque = field(default_factory=lambda: deque(maxlen=5))
    right_ear_history: deque = field(default_factory=lambda: deque(maxlen=5))
    mar_history: deque = field(default_factory=lambda: deque(maxlen=5))
    angle_history: deque = field(default_factory=lambda: deque(maxlen=9))
    side_target: str | None = None
    side_samples: list[tuple[float, float]] = field(default_factory=list)
    side_baselines: dict[str, tuple[float, float]] = field(default_factory=dict)
    head_classifier: HeadPoseClassifier = field(default_factory=HeadPoseClassifier)
    last_status: str | None = None

    @property
    def calibrated(self) -> bool:
        return len(self.left_calibration) >= CALIBRATION_FRAMES

    def start_side_calibration(self, direction: str) -> None:
        if direction not in {"LEFT", "RIGHT"}:
            raise ValueError("side_calibration must be LEFT or RIGHT.")
        self.side_target = direction
        self.side_samples.clear()

    def observe_side(self, pose: str, left_ear: float, right_ear: float) -> None:
        if self.side_target is None or pose != self.side_target:
            return
        self.side_samples.append((left_ear, right_ear))
        if len(self.side_samples) < SIDE_CALIBRATION_FRAMES:
            return
        values = np.asarray(self.side_samples, dtype=float)
        self.side_baselines[self.side_target] = (
            float(np.median(values[:, 0])),
            float(np.median(values[:, 1])),
        )
        self.side_target = None
        self.side_samples.clear()


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


def _head_angles(points, width: int, height: int):
    import cv2

    image_points = np.array(
        [(points[index].x * width, points[index].y * height) for index in POSE_LANDMARKS],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[width, 0.0, width / 2], [0.0, width, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch, yaw, _ = cv2.RQDecomp3x3(rotation_matrix)[0]
    left_outer, right_outer = points[33], points[263]
    roll = degrees(atan2(right_outer.y - left_outer.y, right_outer.x - left_outer.x))
    return float(pitch), float(yaw), float(roll)


def _eye_state(left_ear, right_ear, left_base, right_base, mar, suppress=False):
    ratio = max(left_ear / max(left_base, 1e-8), right_ear / max(right_base, 1e-8))
    if mar > MAR_THRESHOLD:
        return "YAWNING", min(1.0, 0.70 + mar - MAR_THRESHOLD)
    if suppress:
        return "ACTIVE", min(1.0, ratio)
    if ratio < 0.75:
        return "SLEEPING", min(1.0, 0.70 + (0.75 - ratio) / 0.75)
    if ratio < 0.90:
        return "DROWSY", min(1.0, (0.90 - ratio) / 0.15)
    return "ACTIVE", min(1.0, ratio)


def analyze(rgb_frame, session_id, side_calibration=None):
    result = _mesh().process(rgb_frame)
    if not result.multi_face_landmarks:
        return {"detected": False, "message": "No face detected"}
    points = result.multi_face_landmarks[0].landmark
    angles = _head_angles(points, rgb_frame.shape[1], rgb_frame.shape[0])
    if angles is None:
        return {"detected": False, "message": "Head pose was not detected"}
    raw_pitch, raw_yaw, raw_roll = angles
    current_left_ear = _ear(points, LEFT_EYE)
    current_right_ear = _ear(points, RIGHT_EYE)
    current_ear = (current_left_ear + current_right_ear) / 2
    current_mar = _mar(points)
    state = _states.setdefault(str(session_id), SessionState())
    if side_calibration:
        if not state.calibrated:
            raise ValueError("Complete the neutral calibration first.")
        state.start_side_calibration(str(side_calibration).upper())

    if not state.calibrated:
        state.left_calibration.append(current_left_ear)
        state.right_calibration.append(current_right_ear)
        state.pitch_calibration.append(raw_pitch)
        state.yaw_calibration.append(raw_yaw)
        state.roll_calibration.append(raw_roll)
        return {
            "detected": True,
            "calibrating": True,
            "progress": len(state.left_calibration),
            "required": CALIBRATION_FRAMES,
            "ear": round(current_ear, 3),
            "mar": round(current_mar, 3),
            **_landmarks(points),
        }

    base_left = float(np.median(state.left_calibration))
    base_right = float(np.median(state.right_calibration))
    base_pitch = float(np.median(state.pitch_calibration))
    base_yaw = float(np.median(state.yaw_calibration))
    base_roll = float(np.median(state.roll_calibration))
    state.angle_history.append(
        (
            normalize_angle(raw_pitch - base_pitch),
            normalize_angle(raw_yaw - base_yaw),
            normalize_angle(raw_roll - base_roll),
        )
    )
    pitch, yaw, roll = np.median(np.asarray(state.angle_history), axis=0)
    state.left_ear_history.append(current_left_ear)
    state.right_ear_history.append(current_right_ear)
    state.mar_history.append(current_mar)
    left_ear = sum(state.left_ear_history) / len(state.left_ear_history)
    right_ear = sum(state.right_ear_history) / len(state.right_ear_history)
    ear = (left_ear + right_ear) / 2
    mar = sum(state.mar_history) / len(state.mar_history)
    head_pose = state.head_classifier.update(float(pitch), float(roll))
    state.observe_side(head_pose, left_ear, right_ear)
    side_baseline = state.side_baselines.get(head_pose)
    suppress_side_ear = head_pose in {"LEFT", "RIGHT"} and side_baseline is None
    eye_left_base, eye_right_base = side_baseline or (base_left, base_right)
    status, confidence = _eye_state(
        left_ear, right_ear, eye_left_base, eye_right_base, mar, suppress=suppress_side_ear
    )
    changed = status != state.last_status
    state.last_status = status
    return {
        "detected": True,
        "calibrating": False,
        "status": status,
        "confidence": round(confidence, 3),
        "ear": round(ear, 3),
        "left_ear": round(left_ear, 3),
        "right_ear": round(right_ear, 3),
        "mar": round(mar, 3),
        "average_ear": round((base_left + base_right) / 2, 3),
        "head_pose": head_pose,
        "pitch": round(float(pitch), 1),
        "yaw": round(float(yaw), 1),
        "roll": round(float(roll), 1),
        "side_calibration_target": state.side_target,
        "side_calibration_progress": len(state.side_samples),
        "side_calibration_required": SIDE_CALIBRATION_FRAMES,
        "left_side_ready": "LEFT" in state.side_baselines,
        "right_side_ready": "RIGHT" in state.side_baselines,
        "eye_pose_gated": suppress_side_ear,
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
            str(index): [round(points[index].x, 5), round(points[index].y, 5)] for index in used
        },
        "left_eye": LEFT_EYE,
        "right_eye": RIGHT_EYE,
        "mouth": MOUTH,
    }


def clear_session(session_id):
    _states.pop(str(session_id), None)
