import pytest

from apps.driving.fatigue_detector import (
    HeadPoseClassifier,
    SessionState,
    _eye_state,
    classify_head_pose,
    normalize_angle,
)


def test_head_pose_uses_pitch_for_nods_and_roll_for_side_tilts():
    assert classify_head_pose(13, 0) == "FORWARD"
    assert classify_head_pose(-11, 0) == "BACKWARD"
    assert classify_head_pose(0, -21) == "LEFT"
    assert classify_head_pose(0, 21) == "RIGHT"


def test_head_pose_hysteresis_holds_side_pose_until_exit_threshold():
    classifier = HeadPoseClassifier()
    assert classifier.update(0, -21) == "LEFT"
    assert classifier.update(0, -16) == "LEFT"
    assert classifier.update(0, -13) == "NORMAL"


def test_angle_wraparound_supports_backward_pose():
    assert normalize_angle(331) == pytest.approx(-29)


def test_side_calibration_uses_open_eye_median():
    state = SessionState()
    state.start_side_calibration("LEFT")
    for index in range(30):
        state.observe_side("LEFT", 0.30 + index / 1000, 0.32 + index / 1000)
    left_base, right_base = state.side_baselines["LEFT"]
    assert left_base == pytest.approx(0.3145)
    assert right_base == pytest.approx(0.3345)
    assert state.side_target is None


def test_uncalibrated_side_suppresses_ear_but_keeps_yawning():
    assert _eye_state(0.10, 0.10, 0.50, 0.50, 0.10, suppress=True)[0] == "ACTIVE"
    assert _eye_state(0.10, 0.10, 0.50, 0.50, 0.70, suppress=True)[0] == "YAWNING"


def test_calibrated_side_detects_bilateral_closure():
    assert _eye_state(0.20, 0.21, 0.40, 0.42, 0.10)[0] == "SLEEPING"
