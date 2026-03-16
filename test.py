"""
Gesture -> FANUC move (fanucpy) using Ultralytics YOLO Pose.

Gestures:
  Arming: both wrists ABOVE shoulders AND wrists close together
  G1 (pos1): left wrist UP, right wrist DOWN
  G2 (pos2): both wrists UP (and wrists apart, to not collide with arming)
  G3 (pos3): right wrist UP, left wrist DOWN

Timing:
  stabilization: 1.0 s
  armed window:  5.0 s
  cooldown:      0.5 s

Requirements:
  pip install ultralytics opencv-python fanucpy numpy
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
from fanucpy import Robot


# ----------------------------
# CONFIG
# ----------------------------
CAM_INDEX = 0

# Use whatever is available in your Ultralytics installation:
# common options: "yolov8n-pose.pt", newer may be "yolo11n-pose.pt"
MODEL_WEIGHTS = "yolov8n-pose.pt"

# FANUC connection
ROBOT_HOST = "192.168.1.100"
ROBOT_PORT = 18735

# Joint positions (DEGREES) - fill with your real joint targets
POSES_JOINT = {
    "G1": [10, 20, -30, 40, 50, 60],  # Position 1 (left)
    "G2": [11, 21, -31, 41, 51, 61],  # Position 2 (middle)
    "G3": [12, 22, -32, 42, 52, 62],  # Position 3 (right)
}

# Motion tuning
MOVE_VELOCITY = 50
MOVE_ACCEL = 50
CNT_VAL = 0

# Keypoint confidence threshold
CONF_MIN = 0.30

# Gesture thresholds in normalized coords (0..1)
WRISTS_CLOSE_THR = 0.08   # arming: wrists close in X
WRISTS_APART_THR = 0.10   # G2: wrists apart in X (avoid arming collision)

UP_MARGIN = 0.02          # wrist must be at least this much above shoulder (y smaller)
DOWN_MARGIN = 0.02        # wrist must be at least this much below hip (y bigger)

# Timing
STABILIZE_S = 1.0
ARM_WINDOW_S = 5.0
COOLDOWN_S = 0.5

# Debug UI
SHOW_WINDOW = True
WINDOW_NAME = "Pose Gesture Control (ESC to exit)"


# ----------------------------
# COCO keypoint indices (17)
# ----------------------------
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


# ----------------------------
# Helpers
# ----------------------------
def valid(conf: np.ndarray | None, *idxs: int) -> bool:
    """Return True if all required keypoints have enough confidence (or conf is None)."""
    if conf is None:
        return True
    return all(conf[i] >= CONF_MIN for i in idxs)


def is_arming(xyn: np.ndarray, conf: np.ndarray | None) -> bool:
    """
    Arming = both wrists above shoulders + wrists close together.
    """
    need = (L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER)
    if not valid(conf, *need):
        return False

    wl, wr = xyn[L_WRIST], xyn[R_WRIST]
    sl, sr = xyn[L_SHOULDER], xyn[R_SHOULDER]

    left_up = wl[1] < (sl[1] - UP_MARGIN)
    right_up = wr[1] < (sr[1] - UP_MARGIN)
    wrists_close = abs(wl[0] - wr[0]) < WRISTS_CLOSE_THR

    return left_up and right_up and wrists_close


def classify_gesture_3(xyn: np.ndarray, conf: np.ndarray | None) -> str | None:
    """
    Returns "G1", "G2", "G3" or None.
      G1: left up,  right down
      G2: both up (and wrists apart)
      G3: right up, left down
    """
    need = (L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if not valid(conf, *need):
        return None

    wl, wr = xyn[L_WRIST], xyn[R_WRIST]
    sl, sr = xyn[L_SHOULDER], xyn[R_SHOULDER]
    hl, hr = xyn[L_HIP], xyn[R_HIP]

    hip_y = (hl[1] + hr[1]) / 2.0

    left_up = wl[1] < (sl[1] - UP_MARGIN)
    right_up = wr[1] < (sr[1] - UP_MARGIN)

    left_down = wl[1] > (hip_y + DOWN_MARGIN)
    right_down = wr[1] > (hip_y + DOWN_MARGIN)

    wrists_apart = abs(wl[0] - wr[0]) > WRISTS_APART_THR

    # Priority: the asymmetric ones first (most explicit)
    if left_up and right_down:
        return "G1"
    if right_up and left_down:
        return "G3"
    if left_up and right_up and wrists_apart:
        return "G2"

    return None


def get_first_person_keypoints(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns (xyn, conf) for the first detected person, or (None, None) if not available.
    xyn shape: (17,2)
    conf shape: (17,) or None
    """
    if result.keypoints is None:
        return None, None
    if len(result.keypoints.xyn) == 0:
        return None, None

    xyn = result.keypoints.xyn[0].cpu().numpy()
    conf = None
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[0].cpu().numpy()
    return xyn, conf


# ----------------------------
# Main
# ----------------------------
def main():
    # Init robot
    robot = Robot(
        robot_model="Fanuc",
        host=ROBOT_HOST,
        port=ROBOT_PORT,
        ee_DO_type="RDO",
        ee_DO_num=7,
    )
    robot.connect()

    # Init vision
    model = YOLO(MODEL_WEIGHTS)
    cap = cv2.VideoCapture(CAM_INDEX)

    IDLE, ARMED, COOLDOWN = "IDLE", "ARMED", "COOLDOWN"
    state = IDLE
    armed_deadline = 0.0
    cooldown_until = 0.0

    # stabilization tracker
    candidate = None         # "ARM" or "G1"/"G2"/"G3"
    candidate_t = 0.0        # accumulated stable time

    last_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            now = time.time()
            dt = now - last_time
            last_time = now

            # Cooldown handling
            if state == COOLDOWN and now >= cooldown_until:
                state = IDLE
                candidate, candidate_t = None, 0.0

            # Inference
            res = model(frame, verbose=False)[0]
            xyn, conf = get_first_person_keypoints(res)

            # Decide current label depending on state
            label = None
            if xyn is not None:
                if state == IDLE:
                    label = "ARM" if is_arming(xyn, conf) else None
                elif state == ARMED:
                    if now > armed_deadline:
                        state = IDLE
                        candidate, candidate_t = None, 0.0
                        label = None
                    else:
                        label = classify_gesture_3(xyn, conf)
                else:
                    label = None
            else:
                label = None

            # Update stabilization accumulator (time-based)
            if label is None:
                candidate, candidate_t = None, 0.0
            else:
                if label == candidate:
                    candidate_t += dt
                else:
                    candidate, candidate_t = label, 0.0

            # Trigger logic
            if state == IDLE:
                if candidate == "ARM" and candidate_t >= STABILIZE_S:
                    state = ARMED
                    armed_deadline = time.time() + ARM_WINDOW_S
                    candidate, candidate_t = None, 0.0
                    print("ARMED")

            elif state == ARMED:
                if candidate in ("G1", "G2", "G3") and candidate_t >= STABILIZE_S:
                    gesture = candidate
                    joints = POSES_JOINT[gesture]
                    print(f"TRIGGER {gesture} -> joints={joints}")

                    # Execute move (blocking)
                    try:
                        robot.move(
                            "joint",
                            vals=joints,
                            velocity=MOVE_VELOCITY,
                            acceleration=MOVE_ACCEL,
                            cnt_val=CNT_VAL,
                            linear=False,
                        )
                    except Exception as e:
                        print("Robot move failed:", e)

                    # Cooldown
                    state = COOLDOWN
                    cooldown_until = time.time() + COOLDOWN_S
                    candidate, candidate_t = None, 0.0

            # Debug window
            if SHOW_WINDOW:
                overlay = res.plot()
                text1 = f"STATE: {state}"
                text2 = f"CAND: {candidate}  t={candidate_t:.2f}s"
                if state == ARMED:
                    text3 = f"ARMED left: {max(0.0, armed_deadline - time.time()):.1f}s"
                else:
                    text3 = ""

                cv2.putText(overlay, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overlay, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if text3:
                    cv2.putText(overlay, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(WINDOW_NAME, overlay)
                if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                    break

    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        # fanucpy may or may not have disconnect; keep it safe:
        if hasattr(robot, "disconnect"):
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()