import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
from fanucpy import Robot
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PySide6.QtCore import Qt, QThread, QTimer, Signal, QObject, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QAction, QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
ROBOT_CONFIG_PATH = os.path.join(BASE_DIR, "robot_positions.json")

# ============================================================
# CONFIG
# ============================================================
SIM_MODE = True
CAM_INDEX = 0
MIRROR_DISPLAY = True

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1060
CAMERA_REFRESH_MS = 30

TOTAL_PARTS = 5
TARGET_CATEGORIES = ("OK", "REDO1", "REDO2", "NOK")
CATEGORY_COLORS = {
    "OK": (74, 222, 128),
    "REDO1": (250, 204, 21),
    "REDO2": (239, 68, 68),
    "NOK": (59, 130, 246),
    None: (94, 106, 122),
}
CATEGORY_HEX = {
    "OK": "#4ade80",
    "REDO1": "#facc15",
    "REDO2": "#ef4444",
    "NOK": "#3b82f6",
    None: "#5e6a7a",
}
TRAY_CAPACITY_PER_CATEGORY = 5
TRAY_VISUAL_ORDER = ("NOK", "REDO2", "REDO1", "OK")
CATEGORY_UI_LABELS = {
    "OK": "OK",
    "REDO1": "REDO1",
    "REDO2": "REDO2",
    "NOK": "NOK",
}

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

MODE_CLASSIC = "CLASSIC"
MODE_GESTURE = "GESTURE"

APP_TITLE = "FANUC kolaborativní robot - ovládání v Pythonu"
APP_SUBTITLE = "Třídění dávky kolaborativním robotem pomocí gest a klávesnice"

STATE_UI_LABELS = {
    "WAIT_COUNT": "Čeká na počet",
    "WAIT_DIRECTION": "Čeká na kategorii",
    "WAIT_CONFIRM": "Čeká na potvrzení",
    "COOLDOWN": "Prodleva",
    "EXECUTING": "Provádí se",
    "DONE": "Hotovo",
}

MODE_UI_LABELS = {
    "CLASSIC": "Klasický",
    "GESTURE": "Gesta",
}

HAND_UI_LABELS = {
    "Left": "Levá",
    "Right": "Pravá",
}


def ui_state_text(state: str) -> str:
    return STATE_UI_LABELS.get(state, state)


def ui_mode_text(mode: str) -> str:
    return MODE_UI_LABELS.get(mode, mode)


def ui_hand_text(hand: str) -> str:
    return HAND_UI_LABELS.get(hand, hand)

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
        "transfer_joint_velocity": 20,
        "transfer_joint_acceleration": 20,
        "process_joint_velocity": 20,
        "process_joint_acceleration": 20,
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
# LOGGING
# ============================================================
class LogBridge(QObject):
    message_emitted = Signal(str)


class TeeStream:
    def __init__(self, original_stream, bridge: LogBridge):
        self.original_stream = original_stream
        self.bridge = bridge
        self._buffer = ""

    def write(self, text):
        if self.original_stream is not None:
            self.original_stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            cleaned = line.rstrip()
            if cleaned:
                self.bridge.message_emitted.emit(cleaned)

    def flush(self):
        if self.original_stream is not None:
            self.original_stream.flush()


# ============================================================
# CAMERA / MEDIAPIPE SETUP
# ============================================================
def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Kamera otevřena přes CAP_DSHOW, index={index}")
        return cap

    cap.release()
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Kamera otevřena přes CAP_MSMF, index={index}")
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

    validate_robot_config_structure(cfg)
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

    motion = cfg["motion"]
    required_motion_keys = [
        "transfer_joint_velocity",
        "transfer_joint_acceleration",
        "process_joint_velocity",
        "process_joint_acceleration",
        "cnt_val",
        "linear",
        "sleep_after_move_s",
        "sleep_after_gripper_s",
        "home_before_start",
        "home_after_finish",
        "open_gripper_before_pick",
    ]
    for key in required_motion_keys:
        if key not in motion:
            raise ValueError(f"motion must contain: {key}")

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


def get_motion_profile(robot_cfg, profile_name: str):
    motion = robot_cfg["motion"]
    if profile_name == "transfer":
        return {
            "velocity": motion["transfer_joint_velocity"],
            "acceleration": motion["transfer_joint_acceleration"],
        }
    if profile_name == "process":
        return {
            "velocity": motion["process_joint_velocity"],
            "acceleration": motion["process_joint_acceleration"],
        }
    raise ValueError(f"Unknown motion profile: {profile_name}")


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

    transfer_profile = get_motion_profile(robot_cfg, "transfer")
    process_profile = get_motion_profile(robot_cfg, "process")

    print_execution_plan(plan)
    print("SIMULATION OF ROBOT COMMANDS")
    print(
        f"Transfer profile: vel={transfer_profile['velocity']}, "
        f"acc={transfer_profile['acceleration']}"
    )
    print(
        f"Process profile: vel={process_profile['velocity']}, "
        f"acc={process_profile['acceleration']}"
    )
    print(f"CNT value: {motion['cnt_val']}")
    print(f"Initial gripper state: {'OPEN' if gripper_is_open else 'CLOSED'}")

    if motion.get("home_before_start", True):
        print(f"MOVE [transfer] HOME -> {format_joints(home)}")

    for step_idx, step in enumerate(plan, start=1):
        print("-" * 78)
        print(f"STEP {step_idx:02d}")

        if motion.get("open_gripper_before_pick", True):
            maybe_set_gripper(True)

        print(f"MOVE [transfer] source[{step['source_slot']}].approach -> {format_joints(step['source_approach'])}")
        print(f"MOVE [process] source[{step['source_slot']}].pick     -> {format_joints(step['source_pick'])}")
        maybe_set_gripper(False)
        print(f"MOVE [process] source[{step['source_slot']}].approach -> {format_joints(step['source_approach'])}")
        print(f"MOVE [transfer] {step['category']}[{step['target_slot']}].approach -> {format_joints(step['target_approach'])}")
        print(f"MOVE [process] {step['category']}[{step['target_slot']}].place    -> {format_joints(step['target_place'])}")
        maybe_set_gripper(True)
        print(f"MOVE [process] {step['category']}[{step['target_slot']}].approach -> {format_joints(step['target_approach'])}")

    if motion.get("home_after_finish", True):
        print(f"MOVE [transfer] HOME -> {format_joints(home)}")

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

    def update_config(self, robot_cfg):
        self.robot_cfg = robot_cfg
        self.gripper_is_open = self._get_initial_gripper_state()

    def ensure_connected(self):
        if self.connected:
            return

        if not self.robot_cfg.get("configured", False):
            raise RuntimeError(
                "robot_positions.json není označen jako nakonfigurovaný. "
                "Před reálným spuštěním doplň všechny pozice a nastav 'configured' na true."
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
        print(f"Připojeno k robotu FANUC na adrese {conn['host']}:{conn['port']}")
        print(f"Počáteční stav chapadla: {'OTEVŘENO' if self.gripper_is_open else 'ZAVŘENO'}")

    def maybe_disconnect(self):
        if self.robot is not None and hasattr(self.robot, "disconnect"):
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self.connected = False
        self.robot = None
        self.gripper_is_open = self._get_initial_gripper_state()

    def move_joint(self, vals, label: str, profile_name: str):
        motion = self.robot_cfg["motion"]
        profile = get_motion_profile(self.robot_cfg, profile_name)
        print(
            f"MOVE [{profile_name}] {label} -> {format_joints(vals)} "
            f"(vel={profile['velocity']}, acc={profile['acceleration']})"
        )
        self.robot.move(
            "joint",
            vals=vals,
            velocity=profile["velocity"],
            acceleration=profile["acceleration"],
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
            self.move_joint(home, "HOME", "transfer")

        for step_idx, step in enumerate(plan, start=1):
            print("-" * 78)
            print(f"EXECUTING STEP {step_idx:02d}")

            if motion.get("open_gripper_before_pick", True):
                self.set_gripper(True)

            self.move_joint(step["source_approach"], f"source[{step['source_slot']}].approach", "transfer")
            self.move_joint(step["source_pick"], f"source[{step['source_slot']}].pick", "process")
            self.set_gripper(False)
            self.move_joint(step["source_approach"], f"source[{step['source_slot']}].approach", "process")
            self.move_joint(step["target_approach"], f"{step['category']}[{step['target_slot']}].approach", "transfer")
            self.move_joint(step["target_place"], f"{step['category']}[{step['target_slot']}].place", "process")
            self.set_gripper(True)
            self.move_joint(step["target_approach"], f"{step['category']}[{step['target_slot']}].approach", "process")

        if motion.get("home_after_finish", True):
            self.move_joint(home, "HOME", "transfer")


# ============================================================
# DRAWING HELPERS
# ============================================================
def normalized_to_pixel(lm, width: int, height: int):
    x = int(lm.x * width)
    y = int(lm.y * height)
    return x, y


def create_blank_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        ratio = row / max(1, height - 1)
        left = np.array([17, 24, 39], dtype=np.float32)
        right = np.array([9, 64, 97], dtype=np.float32)
        mix = left * (1.0 - ratio) + right * ratio
        base[row, :, :] = mix

    for x in range(0, width, 48):
        cv2.line(base, (x, 0), (x, height), (28, 39, 56), 1)
    for y in range(0, height, 48):
        cv2.line(base, (0, y), (width, y), (28, 39, 56), 1)

    cv2.circle(base, (int(width * 0.12), int(height * 0.18)), 110, (35, 87, 122), -1)
    cv2.circle(base, (int(width * 0.84), int(height * 0.76)), 150, (31, 58, 147), -1)
    base = cv2.GaussianBlur(base, (0, 0), 25)
    return base


def draw_panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color=(12, 18, 30), alpha=0.58):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (67, 89, 118), 1)


def draw_hand(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]

    for a, b in HAND_CONNECTIONS:
        pa = normalized_to_pixel(landmarks[a], w, h)
        pb = normalized_to_pixel(landmarks[b], w, h)
        cv2.line(frame, pa, pb, (74, 222, 128), 2)

    for idx, lm in enumerate(landmarks):
        center = normalized_to_pixel(lm, w, h)
        radius = 6 if idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP) else 4
        color = (255, 191, 0) if idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP) else (120, 196, 255)
        cv2.circle(frame, center, radius, color, -1)


def draw_text_lines(frame: np.ndarray, lines, x: int, y: int, dy: int = 28,
                    font_scale: float = 0.72, color=(238, 244, 255), thickness: int = 2):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def color_for_category(category: Optional[str]):
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS[None])


def frame_to_qpixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


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
        lm_dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP])
        > lm_dist(landmarks[THUMB_IP], landmarks[INDEX_MCP]) + 0.08 * s
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


def count_assignments_by_category(assignments):
    counts = {category: 0 for category in TARGET_CATEGORIES}
    for category in assignments:
        if category in counts:
            counts[category] += 1
    return counts


def check_tray_capacity(tray_counts, batch_counts):
    shortages = {}
    for category in TARGET_CATEGORIES:
        occupied = int(tray_counts.get(category, 0))
        incoming = int(batch_counts.get(category, 0))
        free_slots = max(0, TRAY_CAPACITY_PER_CATEGORY - occupied)
        if incoming > free_slots:
            shortages[category] = {
                "occupied": occupied,
                "incoming": incoming,
                "free": free_slots,
                "capacity": TRAY_CAPACITY_PER_CATEGORY,
            }
    return len(shortages) == 0, shortages


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
# WORKER THREADS
# ============================================================
class RobotExecutionThread(QThread):
    finished_with_result = Signal(bool, str)

    def __init__(self, plan, robot_cfg, robot_controller):
        super().__init__()
        self.plan = plan
        self.robot_cfg = robot_cfg
        self.robot_controller = robot_controller

    def run(self):
        try:
            if SIM_MODE:
                simulate_robot_execution(self.plan, self.robot_cfg)
            else:
                self.robot_controller.execute_plan(self.plan)
            self.finished_with_result.emit(True, "Dávka byla úspěšně provedena.")
        except Exception as exc:
            self.finished_with_result.emit(False, str(exc))


class CurrentJointsThread(QThread):
    finished_with_result = Signal(bool, str)

    def __init__(self, robot_controller):
        super().__init__()
        self.robot_controller = robot_controller

    def run(self):
        try:
            joints = self.robot_controller.get_current_joints()
            self.finished_with_result.emit(True, str(joints))
        except Exception as exc:
            self.finished_with_result.emit(False, str(exc))


# ============================================================
# UI BUILDING BLOCKS
# ============================================================
class CardFrame(QFrame):
    def __init__(self, title: str = "", accent: str = "#38bdf8"):
        super().__init__()
        self.setObjectName("CardFrame")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 18, 18, 18)
        self._layout.setSpacing(12)

        if title:
            title_row = QHBoxLayout()
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background:{accent}; border-radius:5px;")
            title_lbl = QLabel(title)
            title_lbl.setObjectName("CardTitle")
            title_row.addWidget(dot)
            title_row.addWidget(title_lbl)
            title_row.addStretch(1)
            self._layout.addLayout(title_row)

    @property
    def body(self):
        return self._layout


class ClickableModeCard(QFrame):
    clicked = Signal()

    def __init__(self, title: str, description: str, accent: str):
        super().__init__()
        self._accent = accent
        self.setObjectName("ClickableModeCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("ModeCardTitle")
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.description_label = QLabel(description)
        self.description_label.setObjectName("ModeCardDescription")
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self.title_label)
        layout.addWidget(self.description_label)
        layout.addStretch(1)
        self.set_active(False)

    def set_active(self, active: bool):
        border = self._accent if active else "rgba(255,255,255,0.10)"
        self.setStyleSheet(
            f"""
            QFrame#ClickableModeCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(12,18,30,230), stop:1 rgba(18,28,44,230));
                border: 1px solid {border};
                border-radius: 24px;
            }}
            QFrame#ClickableModeCard:hover {{
                border: 1px solid {self._accent};
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16,26,42,245), stop:1 rgba(28,40,62,245));
            }}
            QLabel#ModeCardTitle {{
                color: #f8fbff;
                font-size: 22px;
                font-weight: 800;
            }}
            QLabel#ModeCardDescription {{
                color: #d7e6fb;
                font-size: 15px;
                line-height: 1.5em;
            }}
            """
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)

class MetricValueLabel(QLabel):
    def __init__(self, label: str, value: str = "-"):
        super().__init__()
        self.caption = label
        self.setObjectName("MetricValueLabel")
        self.setWordWrap(True)
        self.update_value(value)

    def update_value(self, value: str):
        self.setText(
            f"<span style='color:#8ea3bb; font-size:11px; letter-spacing:1px;'>{self.caption}</span><br>"
            f"<span style='font-size:18px; color:#f8fbff; font-weight:700;'>{value}</span>"
        )


class AssignmentTile(QFrame):
    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.setObjectName("AssignmentTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.slot_label = QLabel(f"Díl {index}")
        self.slot_label.setObjectName("AssignmentSlotLabel")
        self.value_label = QLabel("—")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setObjectName("AssignmentValueLabel")
        self.value_label.setMinimumHeight(52)

        layout.addWidget(self.slot_label)
        layout.addWidget(self.value_label)
        self.set_category(None)

    def set_category(self, category: Optional[str]):
        hex_color = CATEGORY_HEX.get(category, CATEGORY_HEX[None])
        text = category if category is not None else "NEPŘIŘAZENO"
        self.value_label.setText(text)
        self.value_label.setStyleSheet(
            f"background:{hex_color}; color:#061018; border-radius:12px; font-size:18px; font-weight:800; padding:8px;"
        )


class CategoryRouteCard(QFrame):
    def __init__(self, icon: str, title: str, subtitle: str, accent: str):
        super().__init__()
        self._accent = accent
        self._active = False
        self.setObjectName("CategoryRouteCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        self.icon_label = QLabel(icon)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setObjectName("CategoryIconLabel")
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setObjectName("CategoryTitleLabel")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setObjectName("CategorySubtitleLabel")

        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        self.set_active(False)

    def set_active(self, active: bool):
        self._active = active
        border = self._accent if active else "rgba(255,255,255,0.08)"
        glow = f"rgba({int(self._accent[1:3],16)}, {int(self._accent[3:5],16)}, {int(self._accent[5:7],16)}, 0.14)" if active else "rgba(255,255,255,0.03)"
        self.setStyleSheet(
            f"""
            QFrame#CategoryRouteCard {{
                background: {glow};
                border: 1px solid {border};
                border-radius: 20px;
            }}
            QLabel#CategoryIconLabel {{
                font-size: 34px;
                font-weight: 800;
                color: {self._accent};
            }}
            QLabel#CategoryTitleLabel {{
                font-size: 15px;
                font-weight: 800;
                color: #f8fbff;
            }}
            QLabel#CategorySubtitleLabel {{
                font-size: 12px;
                color: #8ea3bb;
            }}
            """
        )


class TrayDot(QFrame):
    def __init__(self, color_hex: str):
        super().__init__()
        self.color_hex = color_hex
        self.setObjectName("TrayDot")
        self.setFixedSize(26, 26)
        self.set_filled(False)

    def set_filled(self, filled: bool):
        background = self.color_hex if filled else "rgba(255,255,255,0.03)"
        border = self.color_hex if filled else "rgba(255,255,255,0.14)"
        self.setStyleSheet(
            f"background:{background}; border:2px solid {border}; border-radius:13px;"
        )


class TrayRowWidget(QFrame):
    def __init__(self, category: str):
        super().__init__()
        self.category = category
        self.setObjectName("TrayRowWidget")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        self.name_label = QLabel(CATEGORY_UI_LABELS[category])
        self.name_label.setObjectName("TrayRowLabel")
        self.name_label.setMinimumWidth(64)
        self.name_label.setStyleSheet(
            f"color:{CATEGORY_HEX[category]}; font-size:14px; font-weight:800;"
        )

        dots_holder = QWidget()
        dots_holder.setStyleSheet("background: transparent;")
        dots_layout = QHBoxLayout(dots_holder)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        dots_layout.setSpacing(8)
        self.dots = []
        for _ in range(TRAY_CAPACITY_PER_CATEGORY):
            dot = TrayDot(CATEGORY_HEX[category])
            self.dots.append(dot)
            dots_layout.addWidget(dot)
        dots_layout.addStretch(1)

        self.count_label = QLabel(f"0 / {TRAY_CAPACITY_PER_CATEGORY}")
        self.count_label.setObjectName("TrayCountLabel")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.count_label.setMinimumWidth(58)

        layout.addWidget(self.name_label)
        layout.addWidget(dots_holder, stretch=1)
        layout.addWidget(self.count_label)
        self.set_count(0)

    def set_count(self, count: int):
        count = max(0, min(TRAY_CAPACITY_PER_CATEGORY, int(count)))
        for idx, dot in enumerate(self.dots):
            dot.set_filled(idx < count)
        self.count_label.setText(f"{count} / {TRAY_CAPACITY_PER_CATEGORY}")


class TrayCapacityDialog(QDialog):
    def __init__(self, shortages, tray_counts, batch_counts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nedostatek místa v zásobníku")
        self.setModal(True)
        self.resize(720, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        title = QLabel("V zásobníku není dostatek místa")
        title.setObjectName("HeroTitle")
        title.setStyleSheet("font-size:28px;")
        subtitle = QLabel(
            "Kontrola kapacity při potvrzení dávky zjistila, že aktuální zásobník nemá dost volných pozic. "
            "Můžeš dávku zrušit, nebo potvrdit výměnu zásobníku a provést čekající akci."
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        details_card = CardFrame("Kolize kapacity", accent="#38bdf8")
        for category in TRAY_VISUAL_ORDER:
            if category not in shortages:
                continue
            info = shortages[category]
            detail = QLabel(
                f"<b style='color:{CATEGORY_HEX[category]};'>{CATEGORY_UI_LABELS[category]}</b>: "
                f"obsazeno {info['occupied']} / {info['capacity']}, "
                f"v dávce +{info['incoming']}, volno {info['free']}"
            )
            detail.setObjectName("BodyLabel")
            detail.setWordWrap(True)
            details_card.body.addWidget(detail)
        root.addWidget(details_card)

        note = QLabel(
            "Zrušit = vyresetuje aktuální dávku a ponechá stav zásobníku. "
            "Zásobník byl vyměněn = vynuluje digitální dvojče zásobníku a provede potvrzenou dávku."
        )
        note.setObjectName("MutedLabel")
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Zrušit")
        cancel_btn.setObjectName("ActionButton")
        replace_btn = QPushButton("Zásobník byl vyměněn")
        replace_btn.setObjectName("ActionButton")
        cancel_btn.clicked.connect(self.reject)
        replace_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(replace_btn)
        root.addLayout(buttons)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nastavení aplikace")
        self.setModal(True)
        self.resize(720, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        title = QLabel("Nastavení aplikace")
        title.setObjectName("HeroTitle")
        title.setStyleSheet("font-size:28px;")
        subtitle = QLabel("Změna kamery, cesty ke konfiguraci a režimu robota")
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(CAM_INDEX)

        self.config_input = QLineEdit(ROBOT_CONFIG_PATH)
        self.mirror_check = QCheckBox("Zrcadlit obraz")
        self.mirror_check.setChecked(MIRROR_DISPLAY)
        self.sim_check = QCheckBox("Simulační režim")
        self.sim_check.setChecked(SIM_MODE)

        browse_btn = QPushButton("Vybrat JSON")
        browse_btn.setObjectName("ActionButton")
        browse_btn.clicked.connect(self._browse_json)

        grid.addWidget(QLabel("Index kamery"), 0, 0)
        grid.addWidget(self.camera_spin, 0, 1)
        grid.addWidget(QLabel("Cesta ke konfiguračnímu JSONu"), 1, 0)
        grid.addWidget(self.config_input, 1, 1)
        grid.addWidget(browse_btn, 1, 2)
        grid.addWidget(self.mirror_check, 2, 0, 1, 2)
        grid.addWidget(self.sim_check, 3, 0, 1, 2)
        root.addLayout(grid)

        note = QLabel(
            "Nastavení kamery se v režimu gest použije okamžitě. Změna cesty k JSONu vyvolá nové načtení. "
            "Simulační režim ovlivní další provedení dávky."
        )
        note.setObjectName("MutedLabel")
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Zrušit")
        cancel_btn.setObjectName("ActionButton")
        apply_btn = QPushButton("Použít")
        apply_btn.setObjectName("ActionButton")
        cancel_btn.clicked.connect(self.reject)
        apply_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(apply_btn)
        root.addLayout(buttons)

    def _browse_json(self):
        selected, _ = QFileDialog.getOpenFileName(self, "Vybrat konfigurační JSON robota", self.config_input.text(), "Soubory JSON (*.json)")
        if selected:
            self.config_input.setText(selected)

    def values(self):
        return {
            "camera_index": self.camera_spin.value(),
            "config_path": self.config_input.text().strip(),
            "mirror_display": self.mirror_check.isChecked(),
            "sim_mode": self.sim_check.isChecked(),
        }

class ModeSelectionDialog(QDialog):
    def __init__(self, robot_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volba režimu ovládání")
        self.setModal(True)
        self.resize(980, 560)
        self.setMinimumSize(920, 520)
        self.selected_mode = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(20)

        title = QLabel("Zvol režim ovládání")
        title.setObjectName("HeroTitle")
        subtitle = QLabel(
            f"Konfigurace: {os.path.basename(ROBOT_CONFIG_PATH)} | nakonfigurováno={robot_cfg.get('configured', False)} | "
            f"režim robota={'SIMULACE' if SIM_MODE else 'REÁLNÝ ROBOT'}"
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        buttons = QHBoxLayout()
        buttons.setSpacing(18)

        self.classic_btn = ClickableModeCard(
            "Klasický režim",
            "Pouze klávesnice, bez kamery",
            "#38bdf8",
        )
        self.gesture_btn = ClickableModeCard(
            "Režim gest",
            "Kamera + sledování ruky",
            "#4ade80",
        )

        self.classic_btn.clicked.connect(lambda: self._accept_mode(MODE_CLASSIC))
        self.gesture_btn.clicked.connect(lambda: self._accept_mode(MODE_GESTURE))

        buttons.addWidget(self.classic_btn)
        buttons.addWidget(self.gesture_btn)
        root.addLayout(buttons)

        foot = QLabel("K = klasický | G = gesta | Esc = konec")
        foot.setObjectName("MutedLabel")
        root.addWidget(foot)

    def _accept_mode(self, mode: str):
        self.selected_mode = mode
        self.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_K:
            self._accept_mode(MODE_CLASSIC)
            return
        if event.key() == Qt.Key.Key_G:
            self._accept_mode(MODE_GESTURE)
            return
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, selected_mode: str, robot_cfg):
        super().__init__()
        self.selected_mode = selected_mode
        self.robot_cfg = robot_cfg
        self.robot_controller = FanucRobotController(robot_cfg)
        self.rt = reset_runtime()
        self.tray_counts = {category: 0 for category in TARGET_CATEGORIES}
        self.pending_execution_counts = None

        self.cap = None
        self.landmarker = None
        self.execution_thread = None
        self.joints_thread = None
        self._pulse_step = 0

        self.last_frame = create_blank_frame()
        self.last_detected_hand = False
        self.last_current_count = None
        self.last_current_center = None
        self.last_handedness_label = None
        self.last_display_handedness_label = None
        self.last_landmarks = None
        self.last_states = None
        self.last_overlay_hint = ""
        self.last_result = None

        self.log_bridge = LogBridge()
        self.log_bridge.message_emitted.connect(self.append_log)
        self._install_stdout_bridge()

        self.setWindowTitle(APP_TITLE)
        self.resize(1660, 1120)
        self.setMinimumSize(1500, 1080)

        self._build_ui()
        self._create_shortcuts()
        self._setup_mode_resources()
        self._sync_ui()

        self.state_pulse_timer = QTimer(self)
        self.state_pulse_timer.timeout.connect(self._animate_state_glow)
        self.state_pulse_timer.start(140)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.process_tick)
        self.frame_timer.start(CAMERA_REFRESH_MS)

        self.append_log(f"Konfigurace robota načtena z: {ROBOT_CONFIG_PATH}")
        self.append_log(f"SIM_MODE = {SIM_MODE}")
        self.append_log(f"REŽIM OVLÁDÁNÍ = {ui_mode_text(self.selected_mode)}")
        self.centralWidget().setFocus()

    def _install_stdout_bridge(self):
        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr
        sys.stdout = TeeStream(sys.stdout, self.log_bridge)
        sys.stderr = TeeStream(sys.stderr, self.log_bridge)

    def _restore_stdout_bridge(self):
        sys.stdout = self._stdout_original
        sys.stderr = self._stderr_original

    def _build_ui(self):
        central = QWidget()
        central.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(16)

        root.addWidget(self._build_header())

        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(self._build_left_column(), stretch=4)
        content.addWidget(self._build_right_column(), stretch=2)
        root.addLayout(content, stretch=1)

    def _build_header(self):
        card = CardFrame()
        layout = card.body

        row = QHBoxLayout()
        row.setSpacing(18)

        title_col = QVBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("HeroTitle")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("HeroSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        row.addLayout(title_col, stretch=1)

        mode_switch_col = QVBoxLayout()
        mode_switch_col.setSpacing(8)
        mode_label = QLabel("REŽIM OVLÁDÁNÍ")
        mode_label.setObjectName("MutedLabel")
        mode_switch_col.addWidget(mode_label)

        mode_switch_row = QHBoxLayout()
        mode_switch_row.setSpacing(8)
        self.header_classic_btn = self._make_header_mode_button(
            "Klasický",
            "Přepne na ovládání pouze klávesnicí bez restartu aplikace.",
        )
        self.header_gesture_btn = self._make_header_mode_button(
            "Gesta",
            "Přepne na ovládání kamerou a gesty bez restartu aplikace.",
        )
        self.header_classic_btn.clicked.connect(lambda: self.change_input_mode(MODE_CLASSIC))
        self.header_gesture_btn.clicked.connect(lambda: self.change_input_mode(MODE_GESTURE))
        mode_switch_row.addWidget(self.header_classic_btn)
        mode_switch_row.addWidget(self.header_gesture_btn)
        mode_switch_col.addLayout(mode_switch_row)
        row.addLayout(mode_switch_col)

        badge_col = QHBoxLayout()
        badge_col.setSpacing(10)
        self.badge_mode = self._make_badge("REŽIM")
        self.badge_robot = self._make_badge("ROBOT")
        self.badge_state = self._make_badge("STAV")
        self.badge_config = self._make_badge("NASTAVENO")
        for badge in [self.badge_mode, self.badge_robot, self.badge_state, self.badge_config]:
            badge_col.addWidget(badge)
        row.addLayout(badge_col)
        layout.addLayout(row)
        return card

    def _build_left_column(self):
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        camera_card = CardFrame("Živý náhled pracoviště", accent="#38bdf8")
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.video_label.setMinimumWidth(940)
        self.video_label.setMinimumHeight(0)
        self.video_label.setMaximumHeight(16777215)
        self._video_surface_height = None
        self.video_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setObjectName("VideoSurface")
        self.video_label.setToolTip("Živý náhled z kamery v režimu gest, nebo pracovní plocha v klasickém režimu.")
        camera_card.body.addWidget(self.video_label)
        layout.addWidget(camera_card, stretch=5)

        assignment_card = CardFrame("Přiřazení dávky", accent="#4ade80")
        grid = QGridLayout()
        grid.setSpacing(12)
        self.assignment_tiles = []
        for i in range(1, TOTAL_PARTS + 1):
            tile = AssignmentTile(i)
            self.assignment_tiles.append(tile)
            grid.addWidget(tile, 0, i - 1)
        assignment_card.body.addLayout(grid)
        layout.addWidget(assignment_card, stretch=0)
        return wrapper

    def _build_right_column(self):
        scroll = QScrollArea()
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        telemetry = CardFrame("Telemetrie systému", accent="#a78bfa")
        telemetry_grid = QGridLayout()
        telemetry_grid.setHorizontalSpacing(18)
        telemetry_grid.setVerticalSpacing(16)
        self.metric_state = MetricValueLabel("STAV")
        self.metric_next = MetricValueLabel("DALŠÍ POZICE")
        self.metric_count = MetricValueLabel("AKTUÁLNÍ POČET")
        self.metric_locked = MetricValueLabel("UZAMČENÝ POČET")
        self.metric_hand = MetricValueLabel("RUKA")
        self.metric_fingers = MetricValueLabel("T,I,M,R,P")
        self.metric_message = MetricValueLabel("ZPRÁVA")
        self.metric_profiles = MetricValueLabel("POHYB")
        metrics = [
            self.metric_state,
            self.metric_next,
            self.metric_count,
            self.metric_locked,
            self.metric_hand,
            self.metric_fingers,
            self.metric_message,
            self.metric_profiles,
        ]
        for idx, metric in enumerate(metrics):
            telemetry_grid.addWidget(metric, idx // 2, idx % 2)
        telemetry.body.addLayout(telemetry_grid)
        layout.addWidget(telemetry)

        hint_card = CardFrame("Kontextová nápověda", accent="#22d3ee")
        self.hint_label = QLabel()
        self.hint_label.setObjectName("HintLabel")
        self.hint_label.setWordWrap(True)
        hint_card.body.addWidget(self.hint_label)
        layout.addWidget(hint_card)

        tray_card = CardFrame("Digitální dvojče zásobníku", accent="#38bdf8")
        tray_note = QLabel("Aktuální obsazení zásobníku. Kapacita každé barvy je 5 pozic.")
        tray_note.setObjectName("MutedLabel")
        tray_note.setWordWrap(True)
        tray_card.body.addWidget(tray_note)
        self.tray_rows = {}
        for category in TRAY_VISUAL_ORDER:
            row = TrayRowWidget(category)
            self.tray_rows[category] = row
            tray_card.body.addWidget(row)
        layout.addWidget(tray_card)

        routes = CardFrame("Směrování kategorií", accent="#22d3ee")
        route_grid = QGridLayout()
        route_grid.setSpacing(12)
        self.route_cards = {}
        route_specs = [
            ("OK", "→", "OK", "Klávesa 1 / pohyb vpravo"),
            ("REDO1", "↑", "REDO1", "Klávesa 2 / pohyb nahoru"),
            ("REDO2", "↓", "REDO2", "Klávesa 3 / pohyb dolů"),
            ("NOK", "←", "NOK", "Klávesa 4 / pohyb vlevo"),
        ]
        for idx, (key_name, icon, title, subtitle) in enumerate(route_specs):
            card = CategoryRouteCard(icon, title, subtitle, CATEGORY_HEX[key_name])
            self.route_cards[key_name] = card
            route_grid.addWidget(card, idx // 2, idx % 2)
        routes.body.addLayout(route_grid)
        layout.addWidget(routes)

        controls = CardFrame("Akce", accent="#f97316")
        btn_grid = QGridLayout()
        btn_grid.setSpacing(10)
        self.btn_confirm = self._make_action_button("Potvrdit dávku", "Enter / C")
        self.btn_undo = self._make_action_button("Zpět poslední krok", "U")
        self.btn_reset = self._make_action_button("Reset / nová dávka", "R")
        self.btn_reload = self._make_action_button("Načíst JSON znovu", "L")
        self.btn_exit = self._make_action_button("Konec", "Esc")
        self.btn_settings = self._make_action_button("Nastavení", "Ctrl+,")
        actions = [
            (self.btn_confirm, self.handle_confirm),
            (self.btn_undo, self.handle_undo),
            (self.btn_reset, self.handle_reset),
            (self.btn_reload, self.handle_reload_json),
            (self.btn_exit, self.close),
            (self.btn_settings, self.show_settings_dialog),
        ]
        for btn, fn in actions:
            btn.clicked.connect(fn)
        btn_grid.addWidget(self.btn_confirm, 0, 0)
        btn_grid.addWidget(self.btn_undo, 0, 1)
        btn_grid.addWidget(self.btn_reset, 1, 0)
        btn_grid.addWidget(self.btn_reload, 1, 1)
        btn_grid.addWidget(self.btn_exit, 2, 1)
        btn_grid.addWidget(self.btn_settings, 2, 0)
        controls.body.addLayout(btn_grid)
        layout.addWidget(controls)

        log_card = CardFrame("Záznam událostí", accent="#38bdf8")
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_output.setMinimumHeight(260)
        self.log_output.setObjectName("LogOutput")
        log_card.body.addWidget(self.log_output)
        layout.addWidget(log_card)

        legend = CardFrame("Ovládání a nápověda", accent="#4ade80")
        self.legend_label = QLabel()
        self.legend_label.setObjectName("BodyLabel")
        self.legend_label.setWordWrap(True)
        self.legend_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        legend.body.addWidget(self.legend_label)
        layout.addWidget(legend)

        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll


    def _make_badge(self, text: str) -> QLabel:
        badge = QLabel(text)
        badge.setObjectName("BadgeLabel")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setMinimumWidth(120)
        return badge

    def _make_header_mode_button(self, title: str, tooltip: str) -> QPushButton:
        btn = QPushButton(title)
        btn.setCheckable(True)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setObjectName("HeaderToggleButton")
        btn.setMinimumSize(108, 46)
        btn.setToolTip(tooltip)
        return btn

    def _make_action_button(self, title: str, shortcut: str) -> QPushButton:
        btn = QPushButton(f"{title}\n{shortcut}")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(74)
        btn.setObjectName("ActionButton")
        btn.setToolTip(f"{title} ({shortcut})")
        return btn

    def _create_shortcuts(self):
        self._global_shortcuts = []

        for key, fn in [
            ("Return", self.handle_confirm),
            ("Enter", self.handle_confirm),
            ("C", self.handle_confirm),
            ("U", self.handle_undo),
            ("R", self.handle_reset),
            ("L", self.handle_reload_json),
            ("P", self.handle_print_joints),
            ("Escape", self.close),
            ("Ctrl+,", self.show_settings_dialog),
            ("Ctrl+K", lambda: self.change_input_mode(MODE_CLASSIC)),
            ("Ctrl+G", lambda: self.change_input_mode(MODE_GESTURE)),
        ]:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(fn)
            self.addAction(action)


    def _setup_mode_resources(self):
        if self.selected_mode == MODE_GESTURE:
            self.cap = open_camera(CAM_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
            print("Načítá se MediaPipe Hand Landmarker...")
            self.landmarker = create_landmarker(MODEL_PATH)
            print("Hand Landmarker byl načten.")
        else:
            print("Zvolen klasický režim: pouze klávesnice, kamera vypnuta.")

    def _release_mode_resources(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _reset_input_cache(self):
        self.last_detected_hand = False
        self.last_current_count = None
        self.last_current_center = None
        self.last_handedness_label = None
        self.last_display_handedness_label = None
        self.last_landmarks = None
        self.last_states = None
        self.last_result = None

    def _normalize_runtime_after_mode_change(self):
        clear_transient_runtime(self.rt)
        self.rt["stable_count_value"] = None
        if self.rt["state"] not in ("EXECUTING", "DONE"):
            self.rt["state"] = "WAIT_CONFIRM" if all_parts_assigned(self.rt["assignments"]) else "WAIT_COUNT"
        self._reset_input_cache()

    def change_input_mode(self, new_mode: str):
        if new_mode not in (MODE_CLASSIC, MODE_GESTURE):
            return
        if new_mode == self.selected_mode:
            self._sync_ui()
            return

        previous_mode = self.selected_mode

        try:
            if new_mode == MODE_CLASSIC:
                self._release_mode_resources()
                self.selected_mode = MODE_CLASSIC
                print("Zvolen klasický režim: pouze klávesnice, kamera vypnuta.")
            else:
                self.selected_mode = MODE_GESTURE
                self._setup_mode_resources()
        except Exception as exc:
            self.selected_mode = previous_mode
            if previous_mode == MODE_GESTURE and self.cap is None and self.landmarker is None:
                try:
                    self._setup_mode_resources()
                except Exception:
                    pass
            self.show_message(f"Přepnutí režimu selhalo: {exc}", 4.0)
            self.append_log(f"Přepnutí režimu selhalo: {exc}")
            self._sync_ui()
            return

        self._normalize_runtime_after_mode_change()
        self.last_frame = create_blank_frame(1440, 900)
        self.show_message(f"Režim ovládání přepnut na {ui_mode_text(self.selected_mode)}.")
        self.append_log(f"Režim ovládání přepnut na {ui_mode_text(self.selected_mode)}")
        self._sync_ui()

    def _sync_ui(self):
        mode_text = ui_mode_text(self.selected_mode)
        state_text = ui_state_text(self.rt["state"])
        self.badge_mode.setText(f"REŽIM\n{mode_text.upper()}")
        self.badge_robot.setText(f"ROBOT\n{'SIM' if SIM_MODE else 'REÁL'}")
        self.badge_state.setText(f"STAV\n{state_text.upper()}")
        self.badge_config.setText(f"NASTAVENO\n{'Ano' if self.robot_cfg.get('configured', False) else 'Ne'}")
        self.header_classic_btn.setChecked(self.selected_mode == MODE_CLASSIC)
        self.header_gesture_btn.setChecked(self.selected_mode == MODE_GESTURE)

        self.metric_state.update_value(state_text)
        next_slot = f"{self.rt['next_idx'] + 1} / {TOTAL_PARTS}" if self.rt["next_idx"] < TOTAL_PARTS else f"- / {TOTAL_PARTS}"
        self.metric_next.update_value(next_slot)
        self.metric_count.update_value(str(self.last_current_count if self.last_current_count is not None else self.rt["stable_count_value"] or "-"))
        self.metric_locked.update_value(str(self.rt["locked_count"] if self.rt["locked_count"] is not None else "-"))
        self.metric_hand.update_value(ui_hand_text(self.last_display_handedness_label) if self.last_detected_hand else "-")
        self.metric_fingers.update_value(str(self.last_states) if self.last_states is not None else "-")
        self.metric_message.update_value(self.rt["message"] if self.rt["message"] else "Připraveno")

        active_category = None
        if self.rt["history"]:
            active_category = self.rt["history"][-1]["category"]
        if self.rt["state"] == "WAIT_DIRECTION" and self.rt["locked_count"] is not None:
            active_category = None
        for name, card in getattr(self, 'route_cards', {}).items():
            card.set_active(name == active_category)

        motion = self.robot_cfg["motion"]
        motion_text = (
            f"transfer v{motion['transfer_joint_velocity']} / a{motion['transfer_joint_acceleration']}"
            f" | process v{motion['process_joint_velocity']} / a{motion['process_joint_acceleration']}"
        )
        self.metric_profiles.update_value(motion_text)

        for idx, tile in enumerate(self.assignment_tiles):
            tile.set_category(self.rt["assignments"][idx])

        for category, row in getattr(self, 'tray_rows', {}).items():
            row.set_count(self.tray_counts.get(category, 0))

        legend_lines = [
            "<b>Veškerá funkcionalita zůstala zachována.</b>",
            "<br><b>Klávesnice:</b><br>"
            "1..5 = počet dílů<br>"
            "Kategorie 1 = OK<br>"
            "Kategorie 2 = REDO1<br>"
            "Kategorie 3 = REDO2<br>"
            "Kategorie 4 = NOK<br>"
            "Enter / C = potvrdit dávku<br>"
            "U = zpět poslední krok<br>"
            "R = reset aktuální dávky / nová dávka<br>"
            "L = načíst JSON znovu<br>"
            "P = vypsat aktuální kloubové souřadnice robota<br>"
            "Ctrl+K = přepnout na klasický režim<br>"
            "Ctrl+G = přepnout na režim gest<br>"
            "Esc = konec",
        ]
        legend_lines.append(
            "<br><br><b>Logika zásobníku:</b><br>"
            "Kapacita se kontroluje při potvrzení. Pokud je aktuální zásobník plný, aplikace nabídne buď reset dávky, nebo výměnu zásobníku s provedením čekající dávky."
        )
        if self.selected_mode == MODE_GESTURE:
            legend_lines.append(
                "<br><br><b>Režim gest:</b><br>"
                "Podrž kanonické gesto 1..5 pro uzamčení počtu a poté pohybem zvol kategorii.<br>"
                "Gesto potvrzení = (False, False, True, True, True)<br>"
                "Gesto zpět = (True, True, False, False, False)<br>"
                "Gesto nové dávky = (True, False, False, False, True)"
            )
        else:
            legend_lines.append("<br><br><b>Klasický režim:</b><br>Kamera je vypnutá. Celý tok běží pouze přes klávesnici.")
        legend_lines.append(f"<br><br><b>Aktivní JSON:</b><br>{ROBOT_CONFIG_PATH}")
        self.legend_label.setText("".join(legend_lines))
        self._apply_state_glow()

        if self.rt["state"] == "WAIT_COUNT":
            remaining = TOTAL_PARTS - self.rt["next_idx"]
            if self.selected_mode == MODE_GESTURE:
                hint = f"Ukaž kanonické gesto 1..{max(1, remaining)} a drž ruku v klidu."
            else:
                hint = f"Stiskni 1..{max(1, remaining)} pro volbu počtu dílů ve skupině."
        elif self.rt["state"] == "WAIT_DIRECTION":
            if self.selected_mode == MODE_GESTURE:
                hint = f"Uzamčeno {self.rt['locked_count']}. Proveď pohyb VPRAVO / NAHORU / DOLŮ / VLEVO."
            else:
                hint = f"Uzamčeno {self.rt['locked_count']}. Stiskni 1=OK, 2=REDO1, 3=REDO2, 4=NOK."
        elif self.rt["state"] == "WAIT_CONFIRM":
            hint = "Všech 5 dílů je přiřazeno. Potvrď Enter / C nebo podrž gesto potvrzení v režimu gest."
        elif self.rt["state"] == "EXECUTING":
            hint = "Plán robota právě běží. Rozhraní zůstává aktivní a log se dál aktualizuje."
        elif self.rt["state"] == "DONE":
            hint = "Dávka byla provedena. Stiskni R pro novou dávku nebo podrž gesto nové dávky v režimu gest."
        else:
            hint = self.rt["message"] or "Připraveno."
        self.hint_label.setText(hint)

    def _animate_state_glow(self):
        self._pulse_step = (self._pulse_step + 1) % 16
        self._apply_state_glow()

    def _apply_state_glow(self):
        phase = self._pulse_step if self.rt["state"] not in ("WAIT_COUNT", "WAIT_DIRECTION") else 0
        intensity = 0.25 + (phase / 15.0) * 0.55
        state_colors = {
            "WAIT_COUNT": "#38bdf8",
            "WAIT_DIRECTION": "#22d3ee",
            "WAIT_CONFIRM": "#facc15",
            "COOLDOWN": "#a78bfa",
            "EXECUTING": "#fb923c",
            "DONE": "#4ade80",
        }
        accent = state_colors.get(self.rt["state"], "#38bdf8")
        alpha = max(0.20, min(0.85, intensity))
        self.badge_state.setStyleSheet(
            f"background: rgba(18, 28, 44, 0.92); border: 1px solid {accent}; border-radius: 16px; padding: 10px 14px; font-size: 13px; font-weight: 700; color: #d7e6fb;"
        )
        self.metric_state.setStyleSheet(
            f"background: rgba(255,255,255,0.03); border: 2px solid rgba(255,255,255,0.06); border-radius: 18px; padding: 12px;"
            f"border-color: rgba({int(accent[1:3],16)}, {int(accent[3:5],16)}, {int(accent[5:7],16)}, {alpha});"
        )

    def _restart_camera_if_needed(self):
        if self.selected_mode != MODE_GESTURE:
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = open_camera(CAM_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    def show_settings_dialog(self):
        global CAM_INDEX, MIRROR_DISPLAY, SIM_MODE, ROBOT_CONFIG_PATH
        dialog = SettingsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        old_camera = CAM_INDEX
        old_mirror = MIRROR_DISPLAY
        old_sim = SIM_MODE
        old_config = ROBOT_CONFIG_PATH

        CAM_INDEX = values["camera_index"]
        MIRROR_DISPLAY = values["mirror_display"]
        SIM_MODE = values["sim_mode"]
        ROBOT_CONFIG_PATH = values["config_path"] or ROBOT_CONFIG_PATH

        try:
            if self.selected_mode == MODE_GESTURE and (CAM_INDEX != old_camera):
                self._restart_camera_if_needed()
            if ROBOT_CONFIG_PATH != old_config:
                self.handle_reload_json()
            elif old_sim != SIM_MODE or old_mirror != MIRROR_DISPLAY:
                self.show_message("Nastavení bylo použito.")
            self.append_log(
                f"Nastavení aktualizováno | kamera={CAM_INDEX} | zrcadlení={MIRROR_DISPLAY} | sim_režim={SIM_MODE} | json={ROBOT_CONFIG_PATH}"
            )
            self._sync_ui()
        except Exception as exc:
            CAM_INDEX = old_camera
            MIRROR_DISPLAY = old_mirror
            SIM_MODE = old_sim
            ROBOT_CONFIG_PATH = old_config
            self.show_message(f"Použití nastavení selhalo: {exc}", 4.0)
            self.append_log(str(exc))

    def append_log(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {text}")
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def show_message(self, text: str, duration_s: float = MESSAGE_SHOW_S):
        set_ui_message(self.rt, text, duration_s)
        self._sync_ui()

    def process_tick(self):
        try:
            frame, result, detected_hand, current_count, current_center, handedness_label, display_handedness_label, landmarks, states = self._capture_inputs()
            self.last_frame = frame
            self.last_result = result
            self.last_detected_hand = detected_hand
            self.last_current_count = current_count
            self.last_current_center = current_center
            self.last_handedness_label = handedness_label
            self.last_display_handedness_label = display_handedness_label
            self.last_landmarks = landmarks
            self.last_states = states

            self._run_state_machine(detected_hand, current_count, current_center, states)
            self._draw_scene(frame, landmarks, states, current_count)
            self._update_video_surface(frame)
            self._sync_ui()
        except Exception as exc:
            self.show_message(f"Chyba smyčky UI: {exc}", 4.0)
            self.append_log(traceback.format_exc())
            self.frame_timer.stop()

    def _capture_inputs(self):
        if self.selected_mode == MODE_GESTURE:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Nepodařilo se načíst snímek z kamery.")
            if MIRROR_DISPLAY:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            frame = create_blank_frame(1440, 900)
            result = None

        detected_hand = False
        current_count = None
        current_center = None
        handedness_label = None
        display_handedness_label = None
        landmarks = None
        states = None

        if result is not None and result.hand_landmarks:
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

        return frame, result, detected_hand, current_count, current_center, handedness_label, display_handedness_label, landmarks, states

    def _run_state_machine(self, detected_hand, current_count, current_center, states):
        now = time.time()

        if self.rt["message"] and now > self.rt["message_until"]:
            self.rt["message"] = ""
            self.rt["message_until"] = 0.0

        if self.rt["state"] == "COOLDOWN" and now >= self.rt["cooldown_until"]:
            self.rt["state"] = "WAIT_CONFIRM" if all_parts_assigned(self.rt["assignments"]) else "WAIT_COUNT"
            clear_transient_runtime(self.rt)

        if self.selected_mode == MODE_GESTURE:
            self._process_gesture_controls(now, detected_hand, states)

        if self.rt["state"] == "WAIT_COUNT":
            self._state_wait_count(now, detected_hand, current_count, current_center)
        elif self.rt["state"] == "WAIT_DIRECTION":
            self._state_wait_direction(now, detected_hand, current_center)

    def _process_gesture_controls(self, now, detected_hand, states):
        if self.rt["state"] == "DONE" and detected_hand and states is not None:
            if is_new_batch_gesture(states):
                if self.rt["stable_new_batch_started"] is None:
                    self.rt["stable_new_batch_started"] = now
                elif now - self.rt["stable_new_batch_started"] >= NEW_BATCH_HOLD_S:
                    self.rt = reset_runtime()
                    self.show_message("Nová dávka zahájena.")
                    print("Nová dávka zahájena gestem.")
                    return
            else:
                self.rt["stable_new_batch_started"] = None
        elif self.rt["state"] != "DONE":
            self.rt["stable_new_batch_started"] = None

        if self.rt["state"] in ("COOLDOWN", "EXECUTING", "DONE"):
            return

        if detected_hand and states is not None:
            if is_confirm_gesture(states):
                self.rt["stable_undo_started"] = None
                if self.rt["state"] == "WAIT_CONFIRM":
                    if self.rt["stable_confirm_started"] is None:
                        self.rt["stable_confirm_started"] = now
                    elif now - self.rt["stable_confirm_started"] >= CONFIRM_HOLD_S:
                        self.handle_confirm()
                else:
                    self.rt["stable_confirm_started"] = None
                    if not all_parts_assigned(self.rt["assignments"]):
                        self.show_message(f"Potvrzení zablokováno: přiřazeno {self.rt['next_idx']}/{TOTAL_PARTS}")
            elif is_undo_gesture(states):
                self.rt["stable_confirm_started"] = None
                if self.rt["stable_undo_started"] is None:
                    self.rt["stable_undo_started"] = now
                elif now - self.rt["stable_undo_started"] >= UNDO_HOLD_S:
                    self.handle_undo()
            else:
                self.rt["stable_confirm_started"] = None
                self.rt["stable_undo_started"] = None
        else:
            self.rt["stable_confirm_started"] = None
            self.rt["stable_undo_started"] = None

    def _state_wait_count(self, now, detected_hand, current_count, current_center):
        if self.selected_mode != MODE_GESTURE:
            return

        if detected_hand and current_count is not None:
            remaining = TOTAL_PARTS - self.rt["next_idx"]
            if 1 <= current_count <= remaining:
                if self.rt["stable_count_value"] != current_count:
                    self.rt["stable_count_value"] = current_count
                    self.rt["stable_count_started"] = now
                    self.rt["stable_center_ref"] = current_center
                else:
                    if not hand_is_stable(current_center, self.rt["stable_center_ref"], COUNT_STABILITY_CENTER_THR):
                        self.rt["stable_count_started"] = now
                        self.rt["stable_center_ref"] = current_center
                    elif now - self.rt["stable_count_started"] >= COUNT_HOLD_S:
                        self.rt["locked_count"] = current_count
                        self.rt["swipe_start_center"] = current_center
                        self.rt["swipe_started_at"] = now
                        self.rt["state"] = "WAIT_DIRECTION"
                        print(f"LOCKED COUNT = {self.rt['locked_count']}")
            else:
                self.rt["stable_count_value"] = None
                self.rt["stable_count_started"] = None
                self.rt["stable_center_ref"] = None
        else:
            self.rt["stable_count_value"] = None
            self.rt["stable_count_started"] = None
            self.rt["stable_center_ref"] = None

    def _state_wait_direction(self, now, detected_hand, current_center):
        if self.selected_mode != MODE_GESTURE:
            return

        if not detected_hand or current_center is None:
            self.rt["state"] = "WAIT_COUNT"
            self.rt["locked_count"] = None
            self.rt["swipe_start_center"] = None
            self.rt["swipe_started_at"] = None
            return

        if now - self.rt["swipe_started_at"] > SWIPE_WINDOW_S:
            print("Vypršel čas pro pohyb -> návrat do WAIT_COUNT")
            self.rt["state"] = "WAIT_COUNT"
            clear_transient_runtime(self.rt)
            return

        direction = detect_swipe(current_center, self.rt["swipe_start_center"])
        if direction is not None:
            category = CATEGORY_FROM_DIRECTION[direction]
            print(f"POČET={self.rt['locked_count']}, SMĚR={direction}, KATEGORIE={category}")
            self.apply_category_decision(self.rt["locked_count"], category)

    def _draw_scene(self, frame, landmarks, states, current_count):
        if landmarks is not None:
            draw_hand(frame, landmarks)

        if self.selected_mode == MODE_GESTURE and self.rt["state"] == "WAIT_DIRECTION" and self.rt["swipe_start_center"] is not None:
            h, w = frame.shape[:2]
            px = int(self.rt["swipe_start_center"][0] * w)
            py = int(self.rt["swipe_start_center"][1] * h)
            cv2.circle(frame, (px, py), 11, (255, 0, 255), 2)

    def apply_category_decision(self, count: int, category: str):
        now = time.time()
        self.rt["next_idx"] = apply_decision(
            self.rt["assignments"],
            self.rt["history"],
            self.rt["next_idx"],
            count,
            category,
        )

        print("PŘIŘAZENÍ:", assignments_to_string(self.rt["assignments"]))

        if all_parts_assigned(self.rt["assignments"]):
            self.rt["state"] = "WAIT_CONFIRM"
            clear_transient_runtime(self.rt)
            self.show_message("Všech 5 dílů je přiřazeno. Stiskni Enter / C nebo podrž gesto POTVRDIT.", 4.0)
        else:
            self.rt["state"] = "COOLDOWN"
            self.rt["cooldown_until"] = now + COOLDOWN_S
            clear_transient_runtime(self.rt)

    def _handle_classic_category_choice(self, category_number: int):
        if self.selected_mode != MODE_CLASSIC:
            return
        if self.rt["state"] != "WAIT_DIRECTION":
            return

        category_map = {
            1: "OK",
            2: "REDO1",
            3: "REDO2",
            4: "NOK",
        }
        category = category_map.get(category_number)
        if category is None:
            return

        print(f"POČET={self.rt['locked_count']}, KLÁVESA_KATEGORIE={category_number}, KATEGORIE={category}")
        self.apply_category_decision(self.rt["locked_count"], category)

    def _reset_tray_counts(self):
        self.tray_counts = {category: 0 for category in TARGET_CATEGORIES}
        self._sync_ui()

    def _apply_execution_counts_to_tray(self):
        if not self.pending_execution_counts:
            return
        for category, count in self.pending_execution_counts.items():
            self.tray_counts[category] = min(
                TRAY_CAPACITY_PER_CATEGORY,
                self.tray_counts.get(category, 0) + int(count),
            )

    def _start_execution(self, plan, batch_counts, start_message: str = "Provádí se dávka..."):
        self.pending_execution_counts = dict(batch_counts)
        self.rt["state"] = "EXECUTING"
        self.show_message(start_message)
        self.execution_thread = RobotExecutionThread(plan, self.robot_cfg, self.robot_controller)
        self.execution_thread.finished_with_result.connect(self._on_execution_finished)
        self.execution_thread.start()

    def handle_confirm(self):
        if self.execution_thread is not None and self.execution_thread.isRunning():
            return

        if self.rt["state"] != "WAIT_CONFIRM":
            self.show_message(f"Potvrzení zablokováno: přiřazeno {self.rt['next_idx']}/{TOTAL_PARTS}")
            return

        try:
            plan = build_execution_plan(self.rt["assignments"], self.robot_cfg)
        except Exception as exc:
            self.show_message(f"Provedení selhalo: {exc}", 4.0)
            print(f"Provedení selhalo: {exc}")
            return

        batch_counts = count_assignments_by_category(self.rt["assignments"])
        fits, shortages = check_tray_capacity(self.tray_counts, batch_counts)

        if not fits:
            self.append_log("Potvrzení zablokováno: kapacita zásobníku nestačí. Čeká se na rozhodnutí operátora.")
            dialog = TrayCapacityDialog(shortages, self.tray_counts, batch_counts, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._reset_tray_counts()
                self.append_log("Operátor vyměnil zásobník. Stav digitálního dvojčete zásobníku byl vynulován.")
                self._start_execution(plan, batch_counts, "Zásobník vyměněn. Provádí se čekající dávka...")
            else:
                self.rt = reset_runtime()
                self.show_message("Dávka resetována. Aktuální zásobník zůstal beze změny.")
                self.append_log("Operátor zrušil čekající dávku kvůli nedostatečné kapacitě zásobníku.")
                print("Dávka resetována kvůli překročení kapacity zásobníku.")
            return

        self._start_execution(plan, batch_counts)

    def _on_execution_finished(self, ok: bool, message: str):
        clear_transient_runtime(self.rt)
        if ok:
            self._apply_execution_counts_to_tray()
            self.rt["batch_executed"] = True
            self.rt["state"] = "DONE"
            self.show_message(message, 4.0)
        else:
            self.rt["state"] = "WAIT_CONFIRM"
            self.show_message(f"Provedení selhalo: {message}", 4.0)
            print(f"Provedení selhalo: {message}")
        self.pending_execution_counts = None
        self._sync_ui()

    def handle_undo(self):
        if self.rt["history"] and self.rt["state"] not in ("EXECUTING", "DONE"):
            self.rt["next_idx"] = undo_last_step(self.rt["assignments"], self.rt["history"])
            self.rt["state"] = "COOLDOWN"
            self.rt["cooldown_until"] = time.time() + COOLDOWN_S
            clear_transient_runtime(self.rt)
            self.rt["batch_executed"] = False
            self.show_message("Poslední krok byl zrušen.")
            print("Zrušen poslední krok.")
            print("PŘIŘAZENÍ:", assignments_to_string(self.rt["assignments"]))
        else:
            self.show_message("Není co vrátit zpět.")

    def handle_reset(self):
        was_done = self.rt["state"] == "DONE"
        self.rt = reset_runtime()
        self.show_message("Nová dávka zahájena." if was_done else "Dávka resetována.")
        print("Nová dávka zahájena." if was_done else "Dávka resetována.")

    def handle_reload_json(self):
        try:
            self.robot_cfg = load_robot_config(ROBOT_CONFIG_PATH)
            self.robot_controller.maybe_disconnect()
            self.robot_controller = FanucRobotController(self.robot_cfg)
            self.show_message("JSON robota byl načten znovu.")
            print(f"Konfigurace robota byla znovu načtena z: {ROBOT_CONFIG_PATH}")
            self._sync_ui()
        except Exception as exc:
            self.show_message(f"Opětovné načtení selhalo: {exc}", 4.0)
            print(f"Opětovné načtení selhalo: {exc}")

    def handle_print_joints(self):
        if SIM_MODE:
            self.show_message("Výpis kloubů funguje jen v reálném režimu.")
            return
        if self.joints_thread is not None and self.joints_thread.isRunning():
            return
        self.joints_thread = CurrentJointsThread(self.robot_controller)
        self.joints_thread.finished_with_result.connect(self._on_joints_read)
        self.joints_thread.start()

    def _on_joints_read(self, ok: bool, result: str):
        if ok:
            print(f"Aktuální kloubové souřadnice robota: {result}")
            self.show_message(f"Kloubové souřadnice robota byly vypsány do logu: {result}", 4.0)
        else:
            self.show_message(f"Načtení kloubových souřadnic selhalo: {result}", 4.0)
            print(f"Načtení kloubových souřadnic selhalo: {result}")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            event.ignore()
            return

        key = event.key()
        if self.selected_mode == MODE_CLASSIC and self.rt["state"] == "WAIT_COUNT":
            digit_map = {
                Qt.Key.Key_1: 1,
                Qt.Key.Key_2: 2,
                Qt.Key.Key_3: 3,
                Qt.Key.Key_4: 4,
                Qt.Key.Key_5: 5,
            }
            if key in digit_map:
                count = digit_map[key]
                remaining = TOTAL_PARTS - self.rt["next_idx"]
                if 1 <= count <= remaining:
                    self.rt["stable_count_value"] = count
                    self.rt["locked_count"] = count
                    self.rt["state"] = "WAIT_DIRECTION"
                    self.show_message(f"Počet {count} uzamčen. Zvol kategorii pomocí 1..4.")
                    print(f"LOCKED COUNT = {self.rt['locked_count']}")
                else:
                    self.show_message(f"Neplatný počet. Zbývající díly: {remaining}")
                event.accept()
                return

        if self.selected_mode == MODE_CLASSIC and self.rt["state"] == "WAIT_DIRECTION":
            category_digit_map = {
                Qt.Key.Key_1: 1,
                Qt.Key.Key_2: 2,
                Qt.Key.Key_3: 3,
                Qt.Key.Key_4: 4,
            }
            if key in category_digit_map:
                self._handle_classic_category_choice(category_digit_map[key])
                event.accept()
                return

        super().keyPressEvent(event)

    def _update_video_surface(self, frame):
        pixmap = frame_to_qpixmap(frame)
        frame_height, frame_width = frame.shape[:2]

        target_width = max(1, self.video_label.width())
        desired_height = max(240, int(target_width * frame_height / frame_width))

        if self._video_surface_height != desired_height:
            self._video_surface_height = desired_height
            self.video_label.setFixedHeight(desired_height)

        scaled = pixmap.scaled(
            target_width,
            desired_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.last_frame is not None:
            self._update_video_surface(self.last_frame)

    def closeEvent(self, event):
        try:
            if self.frame_timer.isActive():
                self.frame_timer.stop()
            if hasattr(self, 'state_pulse_timer') and self.state_pulse_timer.isActive():
                self.state_pulse_timer.stop()
            self._release_mode_resources()
            if self.execution_thread is not None:
                self.execution_thread.wait(3000)
            if self.joints_thread is not None:
                self.joints_thread.wait(1000)
            self.robot_controller.maybe_disconnect()
        finally:
            self._restore_stdout_bridge()
        super().closeEvent(event)


def build_app_stylesheet() -> str:
    return """
    QWidget {
        background: #070b13;
        color: #f8fbff;
        font-family: 'Segoe UI', 'Inter', 'Arial';
        font-size: 14px;
    }
    QLabel {
        background: transparent;
    }
    QMainWindow {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #060913, stop:0.5 #0a1221, stop:1 #07111f);
    }
    QDialog {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #07111f, stop:1 #09192b);
    }
    QFrame#CardFrame {
        background: rgba(12, 18, 30, 0.90);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
    }
    QLabel#HeroTitle {
        font-size: 34px;
        font-weight: 800;
        color: #f8fbff;
    }
    QLabel#HeroSubtitle {
        font-size: 14px;
        color: #8ea3bb;
    }
    QLabel#CardTitle {
        font-size: 14px;
        font-weight: 700;
        color: #d7e6fb;
        letter-spacing: 1px;
    }
    QLabel#BadgeLabel {
        background: rgba(18, 28, 44, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 700;
        color: #d7e6fb;
    }
    QLabel#BodyLabel {
        color: #d7e6fb;
        line-height: 1.5em;
    }
    QLabel#MutedLabel {
        color: #8ea3bb;
        font-size: 13px;
    }
    QLabel#HintLabel {
        color: #f8fbff;
        font-size: 16px;
        font-weight: 600;
        padding: 4px 0 0 0;
    }
    QLabel#VideoSurface {
        background: #02060d;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    QLabel#MetricValueLabel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 12px;
    }
    QFrame#AssignmentTile {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
    }
    QLabel#AssignmentSlotLabel {
        color: #8ea3bb;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    QLabel#AssignmentValueLabel {
        color: #061018;
        font-size: 18px;
        font-weight: 800;
        border-radius: 12px;
    }
    QFrame#TrayRowWidget {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
    }
    QLabel#TrayCountLabel {
        color: #d7e6fb;
        font-size: 13px;
        font-weight: 700;
    }
    QPushButton#HeaderToggleButton {
        background: rgba(17, 27, 42, 0.96);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        color: #d7e6fb;
        padding: 10px 16px;
        font-size: 13px;
        font-weight: 800;
    }
    QPushButton#HeaderToggleButton:hover {
        border: 1px solid rgba(56, 189, 248, 0.72);
        background: rgba(23, 36, 58, 0.98);
    }
    QPushButton#HeaderToggleButton:checked {
        border: 1px solid #38bdf8;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(15, 31, 52, 0.98), stop:1 rgba(28, 53, 86, 0.98));
        color: #f8fbff;
    }
    QPushButton#ActionButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(19, 32, 54, 0.95), stop:1 rgba(32, 56, 89, 0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        color: #f8fbff;
        padding: 12px;
        font-size: 14px;
        font-weight: 700;
        text-align: center;
    }
    QPushButton#ActionButton:hover {
        border: 1px solid #38bdf8;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(25, 44, 76, 0.98), stop:1 rgba(38, 70, 116, 0.98));
    }
    QPushButton#ActionButton:pressed {
        background: rgba(17, 29, 48, 1.0);
    }
    QPlainTextEdit#LogOutput {
        background: #040811;
        color: #d7e6fb;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 12px;
        selection-background-color: #0ea5e9;
        font-family: Consolas, 'Cascadia Code', monospace;
        font-size: 13px;
    }
    QScrollArea {
        border: none;
        background: transparent;
    }
    QScrollBar:vertical {
        background: transparent;
        width: 12px;
        margin: 4px;
    }
    QScrollBar::handle:vertical {
        background: rgba(255,255,255,0.16);
        min-height: 40px;
        border-radius: 6px;
    }

    QLineEdit, QSpinBox {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 10px 12px;
        color: #f8fbff;
        min-height: 18px;
    }
    QLineEdit:focus, QSpinBox:focus {
        border: 1px solid #38bdf8;
    }
    QCheckBox {
        spacing: 10px;
        color: #d7e6fb;
        font-size: 14px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.16);
        background: rgba(255,255,255,0.04);
    }
    QCheckBox::indicator:checked {
        background: #38bdf8;
        border: 1px solid #38bdf8;
    }
    QToolTip {
        color: #f8fbff;
        background: #07111f;
        border: 1px solid #38bdf8;
        padding: 8px 10px;
    }
    """


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    app.setStyleSheet(build_app_stylesheet())

    try:
        robot_cfg = load_robot_config(ROBOT_CONFIG_PATH)
    except Exception as exc:
        QMessageBox.critical(None, "Chyba konfigurace", str(exc))
        return 1

    dialog = ModeSelectionDialog(robot_cfg)
    if dialog.exec() != QDialog.DialogCode.Accepted or dialog.selected_mode is None:
        return 0

    try:
        window = MainWindow(dialog.selected_mode, robot_cfg)
    except Exception as exc:
        QMessageBox.critical(None, "Chyba při spuštění", str(exc))
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
