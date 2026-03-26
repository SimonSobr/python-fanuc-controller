import json
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fanucpy import Robot

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
ROBOT_CONFIG_PATH = os.path.join(BASE_DIR, "robot_positions.json")

# ============================================================
# CONFIG
# ============================================================
SIM_MODE = False
CAM_INDEX = 0
MIRROR_DISPLAY = True

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

TOTAL_PARTS = 5
TARGET_CATEGORIES = ("OK", "REDO1", "REDO2", "NOK")

COUNT_HOLD_S = 0.7
SWIPE_WINDOW_S = 1.5
COOLDOWN_S = 1.0
CONFIRM_HOLD_S = 1.0
UNDO_HOLD_S = 1.0
NEW_BATCH_HOLD_S = 1.0
MESSAGE_SHOW_S = 2.5

COUNT_STABILITY_CENTER_THR = 0.04
SWIPE_X_THR = 0.15
SWIPE_Y_THR = 0.15
DIRECTION_DOMINANCE = 1.15

MIN_HAND_DET_CONF = 0.8
MIN_HAND_PRESENCE_CONF = 0.8
MIN_TRACKING_CONF = 0.8

CATEGORY_FROM_DIRECTION = {
    "RIGHT": "OK",
    "UP": "REDO1",
    "DOWN": "REDO2",
    "LEFT": "NOK",
}

# ============================================================
# DEFAULT ROBOT CONFIG TEMPLATE
# ============================================================
DEFAULT_ROBOT_CONFIG = {
    "configured": False,
    "robot_connection": {
        "robot_model": "Fanuc",
        "host": "192.168.1.100",
        "port": 18735,
        "ee_DO_type": "RDO",
        "ee_DO_num": 7,
        "use_gripper": True,
        "gripper_toggle_program": "GRIPPER",
        "initial_gripper_open": True,
    },
    "motion": {
        "joint_velocity": 20,
        "joint_acceleration": 20,
        "cnt_val": 0,
        "linear": False,
        "sleep_after_move_s": 0.0,
        "sleep_after_gripper_s": 1.0,
        "home_before_start": True,
        "home_after_finish": True,
        "open_gripper_before_pick": True,
    },
    "poses": {
        "home": [0, 0, 0, 0, 0, 0],
        "sources": {
            "1": {"approach": [0, 0, 0, 0, 0, 0], "pick": [0, 0, 0, 0, 0, 0]},
            "2": {"approach": [0, 0, 0, 0, 0, 0], "pick": [0, 0, 0, 0, 0, 0]},
            "3": {"approach": [0, 0, 0, 0, 0, 0], "pick": [0, 0, 0, 0, 0, 0]},
            "4": {"approach": [0, 0, 0, 0, 0, 0], "pick": [0, 0, 0, 0, 0, 0]},
            "5": {"approach": [0, 0, 0, 0, 0, 0], "pick": [0, 0, 0, 0, 0, 0]},
        },
        "targets": {
            "OK": {
                "1": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "2": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "3": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "4": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "5": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
            },
            "REDO1": {
                "1": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "2": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "3": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "4": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "5": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
            },
            "REDO2": {
                "1": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "2": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "3": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "4": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "5": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
            },
            "NOK": {
                "1": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "2": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "3": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "4": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
                "5": {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]},
            },
        },
    },
}

# ============================================================
# HAND LANDMARK INDICES
# ============================================================
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ============================================================
# CAMERA / MEDIAPIPE SETUP
# ============================================================
def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
        print(f"Camera opened with CAP_DSHOW, index={index}")
        return cap

    cap.release()
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
        print(f"Camera opened with CAP_MSMF, index={index}")
    return cap


def create_landmarker(model_path: str):
    abs_model_path = os.path.abspath(model_path)

    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(
            f"Missing model file: {abs_model_path}\n"
            "Download a compatible MediaPipe Hand Landmarker .task model\n"
            "and save it next to this script."
        )

    size = os.path.getsize(abs_model_path)
    if size < 1_000_000:
        raise RuntimeError(
            f"Model file looks invalid or too small: {abs_model_path} ({size} bytes)"
        )

    base_options = python.BaseOptions(model_asset_path=abs_model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=MIN_HAND_DET_CONF,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )
    return vision.HandLandmarker.create_from_options(options)

# ============================================================
# ROBOT CONFIG / EXECUTION
# ============================================================
def create_default_robot_config_if_missing(config_path: str) -> None:
    if os.path.exists(config_path):
        return

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_ROBOT_CONFIG, f, indent=2)

    print(f"Created default robot config: {config_path}")
    print("Fill in the joint coordinates and set 'configured' to true for real execution.")


def load_robot_config(config_path: str):
    create_default_robot_config_if_missing(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg = migrate_robot_config(cfg)
    validate_robot_config_structure(cfg)
    return cfg



def _blank_target_slots():
    return {
        str(i): {"approach": [0, 0, 0, 0, 0, 0], "place": [0, 0, 0, 0, 0, 0]}
        for i in range(1, TOTAL_PARTS + 1)
    }


def migrate_robot_config(cfg):
    conn = cfg.setdefault("robot_connection", {})
    conn.setdefault("gripper_toggle_program", "GRIPPER")
    conn.setdefault("initial_gripper_open", True)

    motion = cfg.setdefault("motion", {})
    if "sleep_after_gripper_s" not in motion:
        motion["sleep_after_gripper_s"] = 1.0

    poses = cfg.setdefault("poses", {})
    targets = poses.setdefault("targets", {})

    if "REDO" in targets:
        legacy_redo = targets.pop("REDO")
        targets.setdefault("REDO1", legacy_redo)
        targets.setdefault("REDO2", _blank_target_slots())

    return cfg


def validate_joint_list(vals, label: str):
    if not isinstance(vals, list) or len(vals) != 6:
        raise ValueError(f"{label} must be a list of 6 joint values")

    for v in vals:
        if not isinstance(v, (int, float)):
            raise ValueError(f"{label} contains a non-numeric value")


def validate_robot_config_structure(cfg) -> None:
    if "robot_connection" not in cfg or "motion" not in cfg or "poses" not in cfg:
        raise ValueError("robot config must contain: robot_connection, motion, poses")

    poses = cfg["poses"]
    if "home" not in poses or "sources" not in poses or "targets" not in poses:
        raise ValueError("poses must contain: home, sources, targets")

    validate_joint_list(poses["home"], "poses.home")

    for i in range(1, TOTAL_PARTS + 1):
        src = poses["sources"].get(str(i))
        if src is None:
            raise ValueError(f"Missing source slot {i} in robot config")
        validate_joint_list(src.get("approach"), f"poses.sources.{i}.approach")
        validate_joint_list(src.get("pick"), f"poses.sources.{i}.pick")

    for category in TARGET_CATEGORIES:
        cat = poses["targets"].get(category)
        if cat is None:
            raise ValueError(f"Missing target category {category} in robot config")
        for i in range(1, TOTAL_PARTS + 1):
            trg = cat.get(str(i))
            if trg is None:
                raise ValueError(f"Missing target slot {category}.{i} in robot config")
            validate_joint_list(trg.get("approach"), f"poses.targets.{category}.{i}.approach")
            validate_joint_list(trg.get("place"), f"poses.targets.{category}.{i}.place")


def format_joints(vals):
    return "[" + ", ".join(f"{float(v):.3f}" for v in vals) + "]"


def build_execution_plan(assignments, robot_cfg):
    poses = robot_cfg["poses"]
    target_fill_count = {category: 0 for category in TARGET_CATEGORIES}
    plan = []

    for source_slot, category in enumerate(assignments, start=1):
        if category is None:
            raise ValueError("Cannot build execution plan: some parts are still unassigned")

        target_fill_count[category] += 1
        target_slot = target_fill_count[category]

        src = poses["sources"][str(source_slot)]
        trg = poses["targets"][category][str(target_slot)]

        plan.append({
            "source_slot": source_slot,
            "category": category,
            "target_slot": target_slot,
            "source_approach": src["approach"],
            "source_pick": src["pick"],
            "target_approach": trg["approach"],
            "target_place": trg["place"],
        })

    return plan


def print_execution_plan(plan) -> None:
    print("=" * 78)
    print("EXECUTION PLAN")
    for step_idx, step in enumerate(plan, start=1):
        print(
            f"STEP {step_idx:02d}: "
            f"PICK source[{step['source_slot']}] -> "
            f"PLACE {step['category']}[{step['target_slot']}]"
        )
    print("=" * 78)


def simulate_robot_execution(plan, robot_cfg) -> None:
    motion = robot_cfg["motion"]
    home = robot_cfg["poses"]["home"]
    conn = robot_cfg["robot_connection"]
    use_gripper = bool(conn.get("use_gripper", True))
    toggle_program = conn.get("gripper_toggle_program", "GRIPPER")
    toggle_sleep_s = float(motion.get("sleep_after_gripper_s", 1.0))
    gripper_is_open = conn.get("initial_gripper_open", True)

    def maybe_set_gripper(is_open: bool):
        nonlocal gripper_is_open
        if not use_gripper:
            return
        if gripper_is_open == is_open:
            return

        print(f"CALL PROG {toggle_program}  # toggle gripper")
        gripper_is_open = not gripper_is_open
        print("GRIPPER OPEN" if gripper_is_open else "GRIPPER CLOSE")

        if toggle_sleep_s > 0:
            print(f"SLEEP {toggle_sleep_s:.1f}s")

    print_execution_plan(plan)
    print("SIMULATION OF ROBOT COMMANDS")
    print(f"Joint velocity: {motion['joint_velocity']}")
    print(f"Joint acceleration: {motion['joint_acceleration']}")
    print(f"CNT value: {motion['cnt_val']}")
    print(f"Initial gripper state: {'OPEN' if gripper_is_open else 'CLOSED'}")

    if motion.get("home_before_start", True):
        print(f"MOVE HOME -> {format_joints(home)}")

    for step_idx, step in enumerate(plan, start=1):
        print("-" * 78)
        print(f"STEP {step_idx:02d}")

        if motion.get("open_gripper_before_pick", True):
            maybe_set_gripper(True)

        print(f"MOVE source[{step['source_slot']}].approach -> {format_joints(step['source_approach'])}")
        print(f"MOVE source[{step['source_slot']}].pick     -> {format_joints(step['source_pick'])}")
        maybe_set_gripper(False)
        print(f"MOVE source[{step['source_slot']}].approach -> {format_joints(step['source_approach'])}")
        print(f"MOVE {step['category']}[{step['target_slot']}].approach -> {format_joints(step['target_approach'])}")
        print(f"MOVE {step['category']}[{step['target_slot']}].place    -> {format_joints(step['target_place'])}")
        maybe_set_gripper(True)
        print(f"MOVE {step['category']}[{step['target_slot']}].approach -> {format_joints(step['target_approach'])}")

    if motion.get("home_after_finish", True):
        print(f"MOVE HOME -> {format_joints(home)}")

    print("=" * 78)
    print("SIMULATION FINISHED")
    print("=" * 78)


class FanucRobotController:
    def __init__(self, robot_cfg):
        self.robot_cfg = robot_cfg
        self.robot = None
        self.connected = False
        self.gripper_is_open = self._get_initial_gripper_state()

    def _get_initial_gripper_state(self) -> bool:
        return bool(self.robot_cfg["robot_connection"].get("initial_gripper_open", True))

    def _get_gripper_toggle_program(self) -> str:
        return str(self.robot_cfg["robot_connection"].get("gripper_toggle_program", "GRIPPER"))

    def ensure_connected(self):
        if self.connected:
            return

        if not self.robot_cfg.get("configured", False):
            raise RuntimeError(
                "robot_positions.json is not marked as configured. "
                "Fill in all positions and set 'configured' to true before real execution."
            )

        conn = self.robot_cfg["robot_connection"]
        kwargs = {
            "robot_model": conn["robot_model"],
            "host": conn["host"],
            "port": conn["port"],
        }

        if "ee_DO_type" in conn:
            kwargs["ee_DO_type"] = conn["ee_DO_type"]
        if "ee_DO_num" in conn:
            kwargs["ee_DO_num"] = conn["ee_DO_num"]

        self.robot = Robot(**kwargs)
        self.robot.connect()
        self.connected = True
        self.gripper_is_open = self._get_initial_gripper_state()
        print(f"Connected to FANUC robot at {conn['host']}:{conn['port']}")
        print(f"Initial gripper state: {'OPEN' if self.gripper_is_open else 'CLOSED'}")

    def maybe_disconnect(self):
        if self.robot is not None and hasattr(self.robot, "disconnect"):
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self.connected = False
        self.robot = None
        self.gripper_is_open = self._get_initial_gripper_state()

    def move_joint(self, vals, label: str):
        motion = self.robot_cfg["motion"]
        print(f"MOVE {label} -> {format_joints(vals)}")
        self.robot.move(
            "joint",
            vals=vals,
            velocity=motion["joint_velocity"],
            acceleration=motion["joint_acceleration"],
            cnt_val=motion["cnt_val"],
            linear=motion["linear"],
        )
        delay = float(motion.get("sleep_after_move_s", 0.0))
        if delay > 0:
            time.sleep(delay)

    def _toggle_gripper(self):
        program_name = self._get_gripper_toggle_program()
        print(f"CALL PROG {program_name}  # toggle gripper")
        self.robot.call_prog(program_name)
        time.sleep(1.0)
        self.gripper_is_open = not self.gripper_is_open

    def set_gripper(self, is_open: bool):
        use_gripper = bool(self.robot_cfg["robot_connection"].get("use_gripper", True))
        if not use_gripper:
            return

        if self.gripper_is_open == is_open:
            return

        self._toggle_gripper()
        print("GRIPPER OPEN" if self.gripper_is_open else "GRIPPER CLOSE")

        extra_delay = float(self.robot_cfg["motion"].get("sleep_after_gripper_s", 1.0)) - 1.0
        if extra_delay > 0:
            time.sleep(extra_delay)

    def get_current_joints(self):
        self.ensure_connected()
        return self.robot.get_curjpos()

    def execute_plan(self, plan):
        self.ensure_connected()
        motion = self.robot_cfg["motion"]
        home = self.robot_cfg["poses"]["home"]

        print_execution_plan(plan)

        if motion.get("home_before_start", True):
            self.move_joint(home, "HOME")

        for step_idx, step in enumerate(plan, start=1):
            print("-" * 78)
            print(f"EXECUTING STEP {step_idx:02d}")

            if motion.get("open_gripper_before_pick", True):
                self.set_gripper(True)

            self.move_joint(step["source_approach"], f"source[{step['source_slot']}].approach")
            self.move_joint(step["source_pick"], f"source[{step['source_slot']}].pick")
            self.set_gripper(False)
            self.move_joint(step["source_approach"], f"source[{step['source_slot']}].approach")

            self.move_joint(step["target_approach"], f"{step['category']}[{step['target_slot']}].approach")
            self.move_joint(step["target_place"], f"{step['category']}[{step['target_slot']}].place")
            self.set_gripper(True)
            self.move_joint(step["target_approach"], f"{step['category']}[{step['target_slot']}].approach")

        if motion.get("home_after_finish", True):
            self.move_joint(home, "HOME")

# ============================================================
# DRAWING
# ============================================================
def normalized_to_pixel(lm, width: int, height: int):
    x = int(lm.x * width)
    y = int(lm.y * height)
    return x, y


def draw_hand(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]

    for a, b in HAND_CONNECTIONS:
        pa = normalized_to_pixel(landmarks[a], w, h)
        pb = normalized_to_pixel(landmarks[b], w, h)
        cv2.line(frame, pa, pb, (0, 255, 0), 2)

    for idx, lm in enumerate(landmarks):
        center = normalized_to_pixel(lm, w, h)
        radius = 5 if idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP) else 3
        cv2.circle(frame, center, radius, (0, 200, 255), -1)


def put_lines(frame: np.ndarray, lines, x: int = 20, y0: int = 30, dy: int = 28) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (229, 235, 80),
            2,
            cv2.LINE_AA,
        )


def put_lines_right(frame: np.ndarray, lines, right_margin: int = 20, y0: int = 30, dy: int = 28) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    frame_width = frame.shape[1]

    for i, line in enumerate(lines):
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = max(0, frame_width - right_margin - text_width)
        cv2.putText(
            frame,
            line,
            (x, y0 + i * dy),
            font,
            font_scale,
            (229, 235, 80),
            thickness,
            cv2.LINE_AA,
        )

# ============================================================
# HAND / GESTURE LOGIC
# ============================================================
def lm_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def get_palm_scale(landmarks):
    a = lm_dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    b = lm_dist(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
    return max(1e-6, 0.5 * (a + b))


def palm_center(landmarks):
    idxs = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    x = float(sum(landmarks[i].x for i in idxs)) / len(idxs)
    y = float(sum(landmarks[i].y for i in idxs)) / len(idxs)
    return x, y


def hand_is_stable(center_now, center_ref, thr: float) -> bool:
    dx = center_now[0] - center_ref[0]
    dy = center_now[1] - center_ref[1]
    return (dx * dx + dy * dy) ** 0.5 <= thr


def finger_extended(landmarks, tip, pip, mcp):
    s = get_palm_scale(landmarks)

    cond1 = landmarks[tip].y < landmarks[pip].y - 0.10 * s
    cond2 = landmarks[pip].y < landmarks[mcp].y - 0.04 * s
    cond3 = lm_dist(landmarks[tip], landmarks[WRIST]) > lm_dist(landmarks[pip], landmarks[WRIST]) + 0.10 * s

    return cond1 and cond2 and cond3


def thumb_extended(landmarks, handedness_label, mirrored=False):
    s = get_palm_scale(landmarks)

    hand = handedness_label or "Right"
    if mirrored:
        hand = "Left" if hand == "Right" else "Right"

    if hand == "Right":
        cond_side = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x - 0.06 * s
    else:
        cond_side = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x + 0.06 * s

    cond_far = (
        lm_dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) >
        lm_dist(landmarks[THUMB_IP], landmarks[INDEX_MCP]) + 0.08 * s
    )

    return cond_side and cond_far


def finger_states(landmarks, handedness_label, mirrored=False):
    thumb = thumb_extended(landmarks, handedness_label, mirrored=mirrored)
    index = finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle = finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring = finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP)
    pinky = finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    return thumb, index, middle, ring, pinky


def canonical_count_from_states(states):
    if states == (False, True, False, False, False):
        return 1
    if states == (False, True, True, False, False):
        return 2
    if states == (False, True, True, True, False):
        return 3
    if states == (False, True, True, True, True):
        return 4
    if states == (True, True, True, True, True):
        return 5
    return None


def is_confirm_gesture(states):
    return states == (False, False, True, True, True)


def is_undo_gesture(states):
    return states == (True, True, False, False, False)


def is_new_batch_gesture(states):
    return states == (True, False, False, False, True)


def detect_swipe(center_now, center_ref):
    dx = center_now[0] - center_ref[0]
    dy = center_now[1] - center_ref[1]

    if abs(dx) >= SWIPE_X_THR and abs(dx) > abs(dy) * DIRECTION_DOMINANCE:
        return "RIGHT" if dx > 0 else "LEFT"

    if abs(dy) >= SWIPE_Y_THR and abs(dy) > abs(dx) * DIRECTION_DOMINANCE:
        return "UP" if dy < 0 else "DOWN"

    return None

# ============================================================
# ASSIGNMENTS / RUNTIME
# ============================================================
def assignments_to_string(assignments):
    parts = []
    for idx, category in enumerate(assignments, start=1):
        value = category if category is not None else "__"
        parts.append(f"{idx}:{value}")
    return " | ".join(parts)


def all_parts_assigned(assignments) -> bool:
    return all(category is not None for category in assignments)


def apply_decision(assignments, history, next_idx: int, count: int, category: str) -> int:
    start_idx = next_idx
    end_idx = next_idx + count

    for i in range(start_idx, end_idx):
        assignments[i] = category

    history.append({
        "start_idx": start_idx,
        "count": count,
        "category": category,
    })
    return end_idx


def undo_last_step(assignments, history):
    if not history:
        return 0

    last_step = history.pop()
    start_idx = last_step["start_idx"]
    count = last_step["count"]

    for j in range(start_idx, start_idx + count):
        assignments[j] = None

    return start_idx


def set_ui_message(rt, text: str, duration_s: float = MESSAGE_SHOW_S):
    rt["message"] = text
    rt["message_until"] = time.time() + duration_s


def clear_transient_runtime(rt):
    rt["stable_count_value"] = None
    rt["stable_count_started"] = None
    rt["stable_confirm_started"] = None
    rt["stable_undo_started"] = None
    rt["stable_new_batch_started"] = None
    rt["stable_center_ref"] = None
    rt["locked_count"] = None
    rt["swipe_start_center"] = None
    rt["swipe_started_at"] = None


def reset_runtime():
    return {
        "state": "WAIT_COUNT",
        "next_idx": 0,
        "assignments": [None] * TOTAL_PARTS,
        "history": [],
        "stable_count_value": None,
        "stable_count_started": None,
        "stable_confirm_started": None,
        "stable_undo_started": None,
        "stable_new_batch_started": None,
        "stable_center_ref": None,
        "locked_count": None,
        "swipe_start_center": None,
        "swipe_started_at": None,
        "cooldown_until": 0.0,
        "batch_executed": False,
        "message": "",
        "message_until": 0.0,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    robot_cfg = load_robot_config(ROBOT_CONFIG_PATH)
    robot_controller = FanucRobotController(robot_cfg)

    cap = open_camera(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    print("Loading MediaPipe Hand Landmarker...")
    landmarker = create_landmarker(MODEL_PATH)
    print("Hand Landmarker loaded.")
    print(f"Robot config loaded from: {ROBOT_CONFIG_PATH}")
    print(f"SIM_MODE = {SIM_MODE}")

    rt = reset_runtime()
    window_name = "Finger Count + Direction Sorting (OK / REDO1 / REDO2 / NOK)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            if MIRROR_DISPLAY:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            detected_hand = False
            current_count = None
            current_center = None
            handedness_label = None
            display_handedness_label = None
            landmarks = None
            states = None

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                detected_hand = True
                current_center = palm_center(landmarks)

                if result.handedness and result.handedness[0]:
                    handedness_label = result.handedness[0][0].category_name
                else:
                    handedness_label = "Right"

                display_handedness_label = handedness_label
                if MIRROR_DISPLAY:
                    display_handedness_label = "Left" if handedness_label == "Right" else "Right"

                states = finger_states(landmarks, handedness_label, mirrored=MIRROR_DISPLAY)
                current_count = canonical_count_from_states(states)

            now = time.time()

            if rt["message"] and now > rt["message_until"]:
                rt["message"] = ""
                rt["message_until"] = 0.0

            if rt["state"] == "COOLDOWN" and now >= rt["cooldown_until"]:
                rt["state"] = "WAIT_CONFIRM" if all_parts_assigned(rt["assignments"]) else "WAIT_COUNT"
                clear_transient_runtime(rt)

            # ------------------------------------------------
            # NEW BATCH GESTURE (ONLY AFTER DONE)
            # ------------------------------------------------
            if rt["state"] == "DONE" and detected_hand and states is not None:
                if is_new_batch_gesture(states):
                    if rt["stable_new_batch_started"] is None:
                        rt["stable_new_batch_started"] = now
                    elif now - rt["stable_new_batch_started"] >= NEW_BATCH_HOLD_S:
                        rt = reset_runtime()
                        set_ui_message(rt, "New batch started.")
                        print("New batch started by gesture.")
                        continue
                else:
                    rt["stable_new_batch_started"] = None
            else:
                if rt["state"] != "DONE":
                    rt["stable_new_batch_started"] = None

            # ------------------------------------------------
            # GLOBAL CONFIRM / UNDO GESTURES
            # ------------------------------------------------
            if rt["state"] not in ("COOLDOWN", "EXECUTING", "DONE") and detected_hand and states is not None:
                if is_confirm_gesture(states):
                    rt["stable_undo_started"] = None

                    if rt["state"] == "WAIT_CONFIRM":
                        if rt["stable_confirm_started"] is None:
                            rt["stable_confirm_started"] = now
                        elif now - rt["stable_confirm_started"] >= CONFIRM_HOLD_S:
                            try:
                                plan = build_execution_plan(rt["assignments"], robot_cfg)
                                rt["state"] = "EXECUTING"
                                set_ui_message(rt, "Executing batch...")

                                if SIM_MODE:
                                    simulate_robot_execution(plan, robot_cfg)
                                else:
                                    robot_controller.execute_plan(plan)

                                rt["batch_executed"] = True
                                rt["state"] = "DONE"
                                set_ui_message(rt, "Batch executed successfully.", 4.0)
                            except Exception as e:
                                rt["state"] = "WAIT_CONFIRM"
                                set_ui_message(rt, f"Execution failed: {e}", 4.0)
                                print(f"Execution failed: {e}")
                            finally:
                                clear_transient_runtime(rt)
                            continue
                    else:
                        rt["stable_confirm_started"] = None
                        if not all_parts_assigned(rt["assignments"]):
                            set_ui_message(rt, f"Confirm blocked: {rt['next_idx']}/{TOTAL_PARTS} assigned")

                elif is_undo_gesture(states):
                    rt["stable_confirm_started"] = None

                    if rt["stable_undo_started"] is None:
                        rt["stable_undo_started"] = now
                    elif now - rt["stable_undo_started"] >= UNDO_HOLD_S:
                        if rt["history"]:
                            rt["next_idx"] = undo_last_step(rt["assignments"], rt["history"])
                            rt["state"] = "COOLDOWN"
                            rt["cooldown_until"] = now + COOLDOWN_S
                            clear_transient_runtime(rt)
                            rt["batch_executed"] = False
                            set_ui_message(rt, "Last step removed.")
                            print("Undo last step by gesture.")
                            print("ASSIGNMENTS:", assignments_to_string(rt["assignments"]))
                        else:
                            set_ui_message(rt, "Nothing to undo.")
                            rt["stable_undo_started"] = None
                        continue

                else:
                    rt["stable_confirm_started"] = None
                    rt["stable_undo_started"] = None
            else:
                if rt["state"] not in ("COOLDOWN", "EXECUTING", "DONE"):
                    rt["stable_confirm_started"] = None
                    rt["stable_undo_started"] = None

            # ------------------------------------------------
            # WAIT_COUNT
            # ------------------------------------------------
            if rt["state"] == "WAIT_COUNT":
                if detected_hand and current_count is not None:
                    remaining = TOTAL_PARTS - rt["next_idx"]

                    if 1 <= current_count <= remaining:
                        if rt["stable_count_value"] != current_count:
                            rt["stable_count_value"] = current_count
                            rt["stable_count_started"] = now
                            rt["stable_center_ref"] = current_center
                        else:
                            if not hand_is_stable(
                                current_center,
                                rt["stable_center_ref"],
                                COUNT_STABILITY_CENTER_THR,
                            ):
                                rt["stable_count_started"] = now
                                rt["stable_center_ref"] = current_center
                            elif now - rt["stable_count_started"] >= COUNT_HOLD_S:
                                rt["locked_count"] = current_count
                                rt["swipe_start_center"] = current_center
                                rt["swipe_started_at"] = now
                                rt["state"] = "WAIT_DIRECTION"
                                print(f"LOCKED COUNT = {rt['locked_count']}")
                    else:
                        rt["stable_count_value"] = None
                        rt["stable_count_started"] = None
                        rt["stable_center_ref"] = None
                else:
                    rt["stable_count_value"] = None
                    rt["stable_count_started"] = None
                    rt["stable_center_ref"] = None

            # ------------------------------------------------
            # WAIT_DIRECTION
            # ------------------------------------------------
            elif rt["state"] == "WAIT_DIRECTION":
                if not detected_hand or current_center is None:
                    rt["state"] = "WAIT_COUNT"
                    rt["locked_count"] = None
                    rt["swipe_start_center"] = None
                    rt["swipe_started_at"] = None
                else:
                    if now - rt["swipe_started_at"] > SWIPE_WINDOW_S:
                        print("Swipe timeout -> back to WAIT_COUNT")
                        rt["state"] = "WAIT_COUNT"
                        clear_transient_runtime(rt)
                    else:
                        direction = detect_swipe(current_center, rt["swipe_start_center"])
                        if direction is not None:
                            category = CATEGORY_FROM_DIRECTION[direction]
                            print(
                                f"COUNT={rt['locked_count']}, "
                                f"DIRECTION={direction}, "
                                f"CATEGORY={category}"
                            )

                            rt["next_idx"] = apply_decision(
                                rt["assignments"],
                                rt["history"],
                                rt["next_idx"],
                                rt["locked_count"],
                                category,
                            )

                            print("ASSIGNMENTS:", assignments_to_string(rt["assignments"]))

                            if all_parts_assigned(rt["assignments"]):
                                rt["state"] = "WAIT_CONFIRM"
                                clear_transient_runtime(rt)
                                set_ui_message(rt, "All 5 assigned. Hold CONFIRM gesture to execute.", 4.0)
                            else:
                                rt["state"] = "COOLDOWN"
                                rt["cooldown_until"] = now + COOLDOWN_S
                                clear_transient_runtime(rt)

            # ------------------------------------------------
            # WAIT_CONFIRM / EXECUTING / DONE
            # ------------------------------------------------
            elif rt["state"] in ("WAIT_CONFIRM", "EXECUTING", "DONE"):
                pass

            # ------------------------------------------------
            # DRAW UI
            # ------------------------------------------------
            if landmarks is not None:
                draw_hand(frame, landmarks)

            lines = [
                f"STATE: {rt['state']}",
                f"MODE: {'SIMULATION' if SIM_MODE else 'REAL ROBOT'}",
                f"NEXT SLOT: {rt['next_idx'] + 1 if rt['next_idx'] < TOTAL_PARTS else '-'} / {TOTAL_PARTS}",
                f"CURRENT COUNT: {current_count if current_count is not None else '-'}",
                f"LOCKED COUNT: {rt['locked_count'] if rt['locked_count'] is not None else '-'}",
                f"HAND: {display_handedness_label if detected_hand else '-'}",
                f"ASSIGNMENTS: {assignments_to_string(rt['assignments'])}",
                "Directions: RIGHT=OK | UP=REDO1 | DOWN=REDO2 | LEFT=NOK",
                f"Config JSON: {os.path.basename(ROBOT_CONFIG_PATH)} | configured={robot_cfg.get('configured', False)}",
            ]
            key_lines = [
                "C = confirm",
                "R = reset batch",
                "U = undo last step",
                "L = reload JSON",
                "P = print robot joints",
                "ESC = exit",
            ]
            put_lines(frame, lines)
            put_lines_right(frame, key_lines)

            if states is not None:
                if rt["state"] not in ("COOLDOWN", "EXECUTING", "DONE"):
                    if is_confirm_gesture(states):
                        held = 0.0
                        if rt["stable_confirm_started"] is not None:
                            held = now - rt["stable_confirm_started"]

                        cv2.putText(
                            frame,
                            f"CONFIRM gesture ({held:.2f}s / {CONFIRM_HOLD_S:.1f}s)",
                            (20, 474),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                    elif is_undo_gesture(states):
                        held = 0.0
                        if rt["stable_undo_started"] is not None:
                            held = now - rt["stable_undo_started"]

                        cv2.putText(
                            frame,
                            f"UNDO gesture ({held:.2f}s / {UNDO_HOLD_S:.1f}s)",
                            (20, 474),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                elif rt["state"] == "DONE" and is_new_batch_gesture(states):
                    held = 0.0
                    if rt["stable_new_batch_started"] is not None:
                        held = now - rt["stable_new_batch_started"]

                    cv2.putText(
                        frame,
                        f"NEW BATCH gesture ({held:.2f}s / {NEW_BATCH_HOLD_S:.1f}s)",
                        (20, 474),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

            if states is not None:
                cv2.putText(
                    frame,
                    f"T,I,M,R,P = {states}",
                    (20, 514),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            if rt["message"]:
                cv2.putText(
                    frame,
                    rt["message"],
                    (20, 554),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (80, 255, 120),
                    2,
                )

            if rt["state"] == "WAIT_COUNT":
                remaining = TOTAL_PARTS - rt["next_idx"]
                if rt["stable_count_value"] is not None:
                    held = now - rt["stable_count_started"] if rt["stable_count_started"] else 0.0
                    cv2.putText(
                        frame,
                        f"Holding {rt['stable_count_value']} / {remaining} ({held:.2f}s)",
                        (20, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"Show canonical 1..{max(1, remaining)} and hold",
                        (20, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

            elif rt["state"] == "WAIT_DIRECTION":
                cv2.putText(
                    frame,
                    f"Locked {rt['locked_count']}. Swipe RIGHT / UP / DOWN / LEFT",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )
                if rt["swipe_start_center"] is not None:
                    h, w = frame.shape[:2]
                    px = int(rt["swipe_start_center"][0] * w)
                    py = int(rt["swipe_start_center"][1] * h)
                    cv2.circle(frame, (px, py), 10, (255, 0, 255), 2)

            elif rt["state"] == "COOLDOWN":
                pass

            elif rt["state"] == "WAIT_CONFIRM":
                cv2.putText(
                    frame,
                    "All 5 parts assigned - hold CONFIRM gesture or press C",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            elif rt["state"] == "EXECUTING":
                cv2.putText(
                    frame,
                    "Executing robot plan...",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            elif rt["state"] == "DONE":
                cv2.putText(
                    frame,
                    "BATCH EXECUTED - press R or hold NEW BATCH gesture",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key in (ord("r"), ord("R")):
                rt = reset_runtime()
                set_ui_message(rt, "Batch reset.")
                print("Batch reset.")

            elif key in (ord("u"), ord("U")):
                if rt["history"] and rt["state"] not in ("EXECUTING", "DONE"):
                    rt["next_idx"] = undo_last_step(rt["assignments"], rt["history"])
                    rt["state"] = "COOLDOWN"
                    rt["cooldown_until"] = time.time() + COOLDOWN_S
                    clear_transient_runtime(rt)
                    rt["batch_executed"] = False
                    set_ui_message(rt, "Last step removed.")
                    print("Undo last step.")
                    print("ASSIGNMENTS:", assignments_to_string(rt["assignments"]))
                else:
                    set_ui_message(rt, "Nothing to undo.")

            elif key in (ord("c"), ord("C")):
                if rt["state"] == "WAIT_CONFIRM":
                    try:
                        plan = build_execution_plan(rt["assignments"], robot_cfg)
                        rt["state"] = "EXECUTING"
                        set_ui_message(rt, "Executing batch...")

                        if SIM_MODE:
                            simulate_robot_execution(plan, robot_cfg)
                        else:
                            robot_controller.execute_plan(plan)

                        rt["batch_executed"] = True
                        rt["state"] = "DONE"
                        set_ui_message(rt, "Batch executed successfully.", 4.0)
                    except Exception as e:
                        rt["state"] = "WAIT_CONFIRM"
                        set_ui_message(rt, f"Execution failed: {e}", 4.0)
                        print(f"Execution failed: {e}")
                    finally:
                        clear_transient_runtime(rt)
                else:
                    set_ui_message(rt, f"Confirm blocked: {rt['next_idx']}/{TOTAL_PARTS} assigned")

            elif key in (ord("l"), ord("L")):
                try:
                    robot_cfg = load_robot_config(ROBOT_CONFIG_PATH)
                    robot_controller = FanucRobotController(robot_cfg)
                    set_ui_message(rt, "Robot JSON reloaded.")
                    print(f"Robot config reloaded from: {ROBOT_CONFIG_PATH}")
                except Exception as e:
                    set_ui_message(rt, f"Reload failed: {e}", 4.0)
                    print(f"Reload failed: {e}")

            elif key in (ord("p"), ord("P")):
                if SIM_MODE:
                    set_ui_message(rt, "Print joints works only in real mode.")
                else:
                    try:
                        joints = robot_controller.get_current_joints()
                        print(f"Current robot joints: {joints}")
                        set_ui_message(rt, f"Robot joints printed to console: {joints}", 4.0)
                    except Exception as e:
                        set_ui_message(rt, f"Read joints failed: {e}", 4.0)
                        print(f"Read joints failed: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        robot_controller.maybe_disconnect()


if __name__ == "__main__":
    main()