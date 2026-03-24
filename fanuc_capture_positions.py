import copy
import json
import os
import time
from typing import Any, List

from fanucpy import Robot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_CONFIG_PATH = os.path.join(BASE_DIR, "robot_positions.json")
CAPTURE_LOG_PATH = os.path.join(BASE_DIR, "captured_positions_log.jsonl")
BACKUP_SUFFIX = ".backup_before_capture"

DEFAULT_TOTAL_PARTS = 5
DEFAULT_TARGET_CATEGORIES = ("OK", "REDO1", "REDO2", "NOK")


DEFAULT_ROBOT_CONFIG = {
    "configured": False,
    "robot_connection": {
        "robot_model": "Fanuc",
        "host": "192.168.1.100",
        "port": 18735,
        "ee_DO_type": "RDO",
        "ee_DO_num": 7,
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
            str(i): {
                "approach": [0, 0, 0, 0, 0, 0],
                "pick": [0, 0, 0, 0, 0, 0],
            }
            for i in range(1, DEFAULT_TOTAL_PARTS + 1)
        },
        "targets": {
            category: {
                str(i): {
                    "approach": [0, 0, 0, 0, 0, 0],
                    "place": [0, 0, 0, 0, 0, 0],
                }
                for i in range(1, DEFAULT_TOTAL_PARTS + 1)
            }
            for category in DEFAULT_TARGET_CATEGORIES
        },
    },
}


class RobotCaptureTool:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = self.load_or_create_config(config_path)
        self.robot = None
        self.connected = False
        self.backup_created = False

    @staticmethod
    def load_or_create_config(config_path: str) -> dict:
        if not os.path.exists(config_path):
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_ROBOT_CONFIG, f, indent=2)
            print(f"Created default config: {config_path}")
            print("Edit robot_connection if needed, then run the script again.")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "robot_connection" not in cfg or "poses" not in cfg:
            raise ValueError("robot_positions.json must contain 'robot_connection' and 'poses'.")

        return cfg

    def ensure_connected(self) -> None:
        if self.connected:
            return

        conn = self.cfg["robot_connection"]
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
        print(f"Connected to FANUC robot at {conn['host']}:{conn['port']}")

    def disconnect(self) -> None:
        if self.robot is not None and hasattr(self.robot, "disconnect"):
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self.robot = None
        self.connected = False

    def read_current(self):
        self.ensure_connected()
        cur_pose = self.robot.get_curpos()
        cur_jpos = self.robot.get_curjpos()
        return cur_pose, cur_jpos

    def create_backup_once(self) -> None:
        if self.backup_created:
            return
        backup_path = self.config_path + BACKUP_SUFFIX
        if not os.path.exists(backup_path):
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
            print(f"Backup created: {backup_path}")
        self.backup_created = True

    def save_config(self) -> None:
        self.create_backup_once()
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

    def append_log(self, label: str, joints: List[float], pose: List[float], note: str = "") -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "note": note,
            "joints": [round(float(v), 3) for v in joints],
            "pose": [round(float(v), 3) for v in pose],
        }
        with open(CAPTURE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def capture_to_path(self, path: str, note: str = "") -> None:
        cur_pose, cur_jpos = self.read_current()
        joints = [round(float(v), 3) for v in cur_jpos]
        pose = [round(float(v), 3) for v in cur_pose]
        set_nested_value(self.cfg, path, joints)
        self.save_config()
        self.append_log(path, joints, pose, note=note)

        print("-" * 78)
        print(f"SAVED: {path}")
        print(f"JOINTS: {format_vals(joints)}")
        print(f"POSE:   {format_vals(pose)}")
        if note:
            print(f"NOTE:   {note}")
        print(f"Updated JSON: {self.config_path}")
        print(f"Capture log:  {CAPTURE_LOG_PATH}")
        print("-" * 78)


def format_vals(vals: List[float]) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in vals) + "]"


def get_nested_value(obj: Any, path: str):
    ref = obj
    for part in path.split("."):
        ref = ref[part]
    return ref


def set_nested_value(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    ref = obj
    for part in parts[:-1]:
        if part not in ref:
            raise KeyError(f"Missing path segment: {part}")
        ref = ref[part]
    last = parts[-1]
    if last not in ref:
        raise KeyError(f"Missing final path segment: {last}")
    ref[last] = value


def build_guided_paths(cfg: dict) -> List[str]:
    poses = cfg["poses"]
    paths = ["poses.home"]

    sources = poses.get("sources", {})
    for slot in sorted(sources.keys(), key=int):
        paths.append(f"poses.sources.{slot}.approach")
        paths.append(f"poses.sources.{slot}.pick")

    targets = poses.get("targets", {})

    def target_sort_key(category: str):
        preferred = {"OK": 0, "REDO1": 1, "REDO2": 2, "REDO": 3, "NOK": 4}
        return (preferred.get(category, 999), category)

    for category in sorted(targets.keys(), key=target_sort_key):
        for slot in sorted(targets[category].keys(), key=int):
            paths.append(f"poses.targets.{category}.{slot}.approach")
            paths.append(f"poses.targets.{category}.{slot}.place")

    return paths


def ask(prompt: str) -> str:
    return input(prompt).strip()


def print_help() -> None:
    print()
    print("Commands:")
    print("  ENTER  = capture current robot position into the shown JSON path")
    print("  p      = print current robot joints + pose without saving")
    print("  n      = capture with an optional note")
    print("  s      = skip current path")
    print("  b      = go one step back")
    print("  j      = jump to a specific guided index")
    print("  m      = manual capture to any JSON path you type")
    print("  h      = show this help")
    print("  q      = quit")
    print()


def run_guided_capture(tool: RobotCaptureTool) -> None:
    paths = build_guided_paths(tool.cfg)
    idx = 0
    total = len(paths)

    print("Guided capture mode")
    print(f"Total positions in current JSON: {total}")
    print_help()

    while 0 <= idx < total:
        path = paths[idx]
        current_value = get_nested_value(tool.cfg, path)

        print("=" * 78)
        print(f"[{idx + 1}/{total}] {path}")
        print(f"Current stored value: {format_vals(current_value)}")
        print("Move robot with teach pendant, then choose an action.")

        cmd = ask("ENTER/p/n/s/b/j/m/h/q > ").lower()

        if cmd == "":
            tool.capture_to_path(path)
            idx += 1

        elif cmd == "p":
            cur_pose, cur_jpos = tool.read_current()
            print(f"Current joints: {format_vals(cur_jpos)}")
            print(f"Current pose:   {format_vals(cur_pose)}")

        elif cmd == "n":
            note = ask("Optional note: ")
            tool.capture_to_path(path, note=note)
            idx += 1

        elif cmd == "s":
            print(f"Skipped: {path}")
            idx += 1

        elif cmd == "b":
            idx = max(0, idx - 1)

        elif cmd == "j":
            raw = ask(f"Jump to step 1-{total}: ")
            try:
                jump_to = int(raw)
                if 1 <= jump_to <= total:
                    idx = jump_to - 1
                else:
                    print("Invalid step number.")
            except ValueError:
                print("Invalid number.")

        elif cmd == "m":
            run_manual_capture(tool)

        elif cmd == "h":
            print_help()

        elif cmd == "q":
            break

        else:
            print("Unknown command. Press h for help.")

    print("Guided capture finished.")


def run_manual_capture(tool: RobotCaptureTool) -> None:
    print()
    print("Manual capture mode")
    print("Example path: poses.sources.1.approach")
    print("Example path: poses.targets.OK.3.place")
    print("Leave empty to return.")

    while True:
        path = ask("JSON path > ")
        if not path:
            print("Leaving manual mode.")
            print()
            return

        try:
            current_value = get_nested_value(tool.cfg, path)
        except Exception as e:
            print(f"Invalid path: {e}")
            continue

        print(f"Current stored value: {format_vals(current_value)}")
        note = ask("Optional note: ")
        confirm = ask("Capture current robot position here? [y/N]: ").lower()
        if confirm == "y":
            tool.capture_to_path(path, note=note)
        else:
            print("Cancelled.")


def main() -> None:
    print("FANUC position capture tool")
    print(f"Config JSON: {ROBOT_CONFIG_PATH}")
    print(f"Capture log: {CAPTURE_LOG_PATH}")
    print()

    tool = RobotCaptureTool(ROBOT_CONFIG_PATH)

    try:
        while True:
            mode = ask("Choose mode: [g]uided, [m]anual, [q]uit > ").lower()
            if mode in ("", "g"):
                run_guided_capture(tool)
                break
            if mode == "m":
                run_manual_capture(tool)
            elif mode == "q":
                break
            else:
                print("Unknown option.")
    finally:
        tool.disconnect()


if __name__ == "__main__":
    main()
