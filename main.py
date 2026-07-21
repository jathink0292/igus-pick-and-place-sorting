import cv2
import igus
import json
import threading
import time
import numpy as np
import paho.mqtt.client as mqtt
import random
import string
import winsound
import math
import subprocess
from vision import BoxDetector, Box

# ============================================================================
# CONFIGURE THESE
# ============================================================================

ROBOT_HOST = "192.168.3.11"
ROBOT_PORT = 3920
MQTT_HOST  = "mqtt-dashboard.com"
MQTT_PORT  = 1883

ROBOT_NUMBER    = 4
TOPIC_BASE      = f"IGUS/robot{ROBOT_NUMBER}"
TOPIC_SORT      = TOPIC_BASE + "/sort"
TOPIC_STATUS    = TOPIC_BASE + "/status"

PICK_TABLE_X_OFFSET    = 0.0
PICK_TABLE_Y_OFFSET    = 0.0
STORAGE_TABLE_X_OFFSET = 0.0
STORAGE_TABLE_Y_OFFSET = 0.0

PICK_Z  = 94
PLACE_Z = 121.1

TRAVEL_Z            = 250
BOX_HEIGHT          = 50.0
SUCTION_ON_CHANNEL  = 31
SUCTION_OFF_CHANNEL = 30
ORIENT              = (180.0, 0.0, 180.0)
FACE_AWAY_POSE      = igus.Joint(A1=90.0)
GRIPPER_X_OFFSET    = 8.0
GRIPPER_Y_OFFSET    = 4.0
SETTLE_TIME         = 2.0

# ============================================================================
# VOICE SETUP
# ============================================================================

def say(phrase: str):
    """
    Speaks using Windows built-in PowerShell speech synthesis.
    Runs in background thread so robot never waits for speech to finish.
    No extra libraries needed — works on all Windows PCs.
    """
    print(f"[Voice] {phrase}")
    safe = phrase.replace("'", "").replace('"', "")
    threading.Thread(
        target=lambda: subprocess.run(
            ["powershell", "-Command",
             f"Add-Type -AssemblyName System.Speech; "
             f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             f"$s.Rate = 2; "
             f"$s.Speak('{safe}')"],
            capture_output=True
        ),
        daemon=True
    ).start()
    time.sleep(0.5)

# ============================================================================
# ROBOT SETUP
# ============================================================================

robot = igus.IGUS(host=ROBOT_HOST, port=ROBOT_PORT, name="IGUS REBEL (physical)")
robot.wait = True
robot.connect()
say("Robot connected and ready.")

_dout_cmd_id = igus.CommandID()

def _set_dout(channel, state):
    state_str = "true" if state else "false"
    robot.send(f"CRISTART {_dout_cmd_id.get_id()} CMD DOUT {channel} {state_str} CRIEND")

def gripper_open():
    _set_dout(SUCTION_OFF_CHANNEL, True)
    _set_dout(SUCTION_ON_CHANNEL, False)
    print("[Gripper] SUCTION OFF (release)")
    time.sleep(SETTLE_TIME)

def gripper_close():
    _set_dout(SUCTION_OFF_CHANNEL, False)
    _set_dout(SUCTION_ON_CHANNEL, True)
    print("[Gripper] SUCTION ON (grab)")
    time.sleep(SETTLE_TIME)

# ============================================================================
# MQTT SETUP
# ============================================================================

_client_id      = "app-" + "".join(random.choices(string.ascii_letters + string.digits, k=12))
mqtt_client     = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                               client_id=_client_id, clean_session=True)
_storage_x      = 0.0
_storage_y      = 0.0
_sort_requested = threading.Event()
_mqtt_connected = [False]

def publish_status(message):
    print(f"[MQTT OUT] {message}")
    mqtt_client.publish(TOPIC_STATUS, message)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        _mqtt_connected[0] = True
        print(f"[MQTT] Connected to {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(TOPIC_SORT)
        say("M Q T T connected.")
    else:
        _mqtt_connected[0] = False

def on_disconnect(client, userdata, rc):
    _mqtt_connected[0] = False
    print(f"[MQTT] Disconnected: rc={rc}")

def on_message(client, userdata, msg):
    global _storage_x, _storage_y
    payload = msg.payload.decode("utf-8")
    print(f"[MQTT IN]  {msg.topic}  ->  {payload}")
    if msg.topic == TOPIC_SORT:
        try:
            data       = json.loads(payload)
            _storage_x = float(data["X"])
            _storage_y = float(data["Y"])
            print(f"[SORT] Storage spot: X={_storage_x} Y={_storage_y} mm")
            say("Start command received. Beginning sorting sequence.")
            _sort_requested.set()
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"[SORT] Bad payload: {e}")
            say("Error. Invalid start message.")

mqtt_client.on_connect    = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message    = on_message
mqtt_client.connect(MQTT_HOST, MQTT_PORT)
mqtt_client.loop_start()

# ============================================================================
# VISION SETUP
# ============================================================================

detector = BoxDetector()
WIN = "Autonomous Sorting - Live View"

# ============================================================================
# DASHBOARD + TRAJECTORY STATE
# ============================================================================

_dash_lock = threading.Lock()
_dash = {
    "status":      "IDLE",
    "detected":    0,
    "picked":      0,
    "total":       0,
    "current_box": "-",
    "travel_m":    0.0,
    "start_time":  None,
    "robot_ok":    True,
}

_traj_lock  = threading.Lock()
_trajectory = []
_prev_pos   = [None]

DASH_W = 320

def _draw_dashboard(panel_h):
    panel = np.zeros((panel_h, DASH_W, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    with _dash_lock:
        status      = _dash["status"]
        detected    = _dash["detected"]
        picked      = _dash["picked"]
        total       = _dash["total"]
        current_box = _dash["current_box"]
        travel_m    = _dash["travel_m"]
        start_time  = _dash["start_time"]
        robot_ok    = _dash["robot_ok"]

    mqtt_ok = _mqtt_connected[0]

    def txt(text, y, color=(255,255,255), scale=0.55, thick=1):
        cv2.putText(panel, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def divider(y):
        cv2.line(panel, (8, y), (DASH_W-8, y), (60,60,60), 1)

    cv2.rectangle(panel, (0,0), (DASH_W, 42), (0,80,0), -1)
    txt("IGUS AUTONOMOUS SORTER", 16, (255,255,255), 0.5, 1)
    txt(f"Robot {ROBOT_NUMBER}  |  SRH Berlin", 34, (180,255,180), 0.38, 1)

    y = 60
    divider(y); y += 18
    txt("Robot Status", y, (180,180,180), 0.42); y += 22

    if status == "IDLE":
        dot_col = (100,100,100); status_col = (160,160,160)
    elif status in ("DETECTING","RUNNING"):
        dot_col = (0,220,0);     status_col = (0,255,0)
    else:
        dot_col = (0,200,255);   status_col = (0,220,255)

    cv2.circle(panel, (18, y-4), 7, dot_col, -1)
    txt(f"  {status}", y, status_col, 0.65, 2); y += 28

    divider(y); y += 18
    txt("Detected Boxes", y, (180,180,180), 0.42); y += 22
    txt(str(detected), y, (255,255,100), 0.9, 2); y += 32

    divider(y); y += 18
    txt("Picked", y, (180,180,180), 0.42); y += 22
    bar_w = DASH_W - 24
    cv2.rectangle(panel, (12,y), (12+bar_w, y+18), (50,50,50), -1)
    if total > 0:
        fill = int(bar_w * picked / total)
        cv2.rectangle(panel, (12,y), (12+fill, y+18), (0,200,80), -1)
    pct = int(100*picked/total) if total > 0 else 0
    txt(f"{picked}/{total}  {pct}%", y+14, (255,255,255), 0.42)
    y += 30

    divider(y); y += 18
    txt("Current Box", y, (180,180,180), 0.42); y += 22
    txt(current_box, y, (0,220,255), 0.75, 2); y += 32

    divider(y); y += 18
    txt("Travel Distance", y, (180,180,180), 0.42); y += 22
    txt(f"{travel_m:.2f} m", y, (255,200,100), 0.75, 2); y += 32

    divider(y); y += 18
    txt("Runtime", y, (180,180,180), 0.42); y += 22
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        m, s = divmod(elapsed, 60)
        runtime_str = f"{m:02d}:{s:02d} sec"
    else:
        runtime_str = "00:00 sec"
    txt(runtime_str, y, (200,200,255), 0.75, 2); y += 32

    divider(y); y += 18
    txt("MQTT", y, (180,180,180), 0.42); y += 22
    if mqtt_ok:
        cv2.circle(panel, (18,y-4), 6, (0,220,0), -1)
        txt("  CONNECTED", y, (0,255,0), 0.55)
    else:
        cv2.circle(panel, (18,y-4), 6, (0,0,200), -1)
        txt("  DISCONNECTED", y, (0,80,255), 0.55)
    y += 26

    divider(y); y += 18
    txt("Robot", y, (180,180,180), 0.42); y += 22
    if robot_ok:
        cv2.circle(panel, (18,y-4), 6, (0,220,0), -1)
        txt("  CONNECTED", y, (0,255,0), 0.55)
    else:
        cv2.circle(panel, (18,y-4), 6, (0,0,200), -1)
        txt("  DISCONNECTED", y, (0,80,255), 0.55)
    y += 26

    divider(panel_h - 28)
    txt("SRH University Berlin", panel_h-14, (80,80,80), 0.38)
    return panel


def _draw_trajectory(frame):
    out = frame
    with _traj_lock:
        pts = list(_trajectory)
    if len(pts) < 1:
        return out
    for i in range(1, len(pts)):
        cv2.line(out, (pts[i-1][0], pts[i-1][1]),
                      (pts[i][0],   pts[i][1]),
                 (0,180,255), 2, cv2.LINE_AA)
    for px, py, label, color in pts:
        cv2.circle(out, (px,py), 7, color, -1)
        cv2.circle(out, (px,py), 7, (255,255,255), 1)
        cv2.putText(out, label, (px+10, py-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out

# ============================================================================
# CAMERA THREADS
# ============================================================================

_frame_lock   = threading.Lock()
_latest_frame = [None]

_display_lock = threading.Lock()
_disp_boxes   = [[]]
_disp_active  = [None]
_disp_storage = [None]

_stop_event   = threading.Event()


def camera_read_thread():
    cap = detector.cap
    while not _stop_event.is_set():
        try:
            ok, frame = cap.read()
            if ok and frame is not None:
                with _frame_lock:
                    _latest_frame[0] = frame.copy()
            time.sleep(0.01)
        except Exception as e:
            print(f"[Camera Read] Error: {e}")
            time.sleep(0.1)


def camera_display_thread():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    while not _stop_event.is_set():
        try:
            with _frame_lock:
                frame = _latest_frame[0]
            if frame is not None:
                with _display_lock:
                    boxes         = list(_disp_boxes[0])
                    active_id     = _disp_active[0]
                    storage_pixel = _disp_storage[0]
                vis      = detector.draw(frame, boxes,
                                         active_id=active_id,
                                         storage_pixel=storage_pixel)
                vis      = _draw_trajectory(vis)
                h, w     = vis.shape[:2]
                dash     = _draw_dashboard(h)
                combined = np.hstack([vis, dash])
                cv2.imshow(WIN, combined)
            cv2.waitKey(30)
            time.sleep(0.01)
        except Exception as e:
            print(f"[Camera Display] Error: {e}")
            time.sleep(0.1)


def set_display(boxes=None, active_id=None, storage_pixel=None):
    with _display_lock:
        if boxes is not None:
            _disp_boxes[0]   = boxes
        if active_id is not None:
            _disp_active[0]  = active_id
        if storage_pixel is not None:
            _disp_storage[0] = storage_pixel


def clear_active():
    with _display_lock:
        _disp_active[0] = None


def get_fresh_frame():
    time.sleep(0.2)
    with _frame_lock:
        frame = _latest_frame[0]
    if frame is None:
        raise RuntimeError("No camera frame available yet.")
    return frame.copy()


def robot_to_pixel(rx, ry):
    M  = detector.transform
    A2 = M[:, :2]
    b  = np.array([rx - M[0, 2], ry - M[1, 2]], dtype=np.float32)
    uv = np.linalg.solve(A2, b)
    return int(uv[0]), int(uv[1])


def add_trajectory_point(robot_x, robot_y, label, color=(0,180,255)):
    try:
        px, py = robot_to_pixel(robot_x, robot_y)
        with _traj_lock:
            _trajectory.append((px, py, label, color))
    except Exception:
        pass


def clear_trajectory():
    with _traj_lock:
        _trajectory.clear()


def update_travel(dx_mm, dy_mm):
    dist_m = math.sqrt(dx_mm**2 + dy_mm**2) / 1000.0
    with _dash_lock:
        _dash["travel_m"] += dist_m


def update_dash(**kwargs):
    with _dash_lock:
        for k, v in kwargs.items():
            if k in _dash:
                _dash[k] = v

# ============================================================================
# START CAMERA THREADS
# ============================================================================

t_read    = threading.Thread(target=camera_read_thread,
                              daemon=True, name="CameraRead")
t_display = threading.Thread(target=camera_display_thread,
                              daemon=True, name="CameraDisplay")
t_read.start()
t_display.start()
print("[Camera] Read and display threads started.")
time.sleep(1.0)

# ============================================================================
# PICK-AND-PLACE MOTION
# ============================================================================

def _move(source: igus.Cart, destination: igus.Cart,
          box_label="", storage_label="Storage"):
    src_above = igus.Cart(source.X,      source.Y,      TRAVEL_Z, *ORIENT)
    dst_above = igus.Cart(destination.X, destination.Y, TRAVEL_Z, *ORIENT)

    print(f"[MOVE] src=({source.X:.0f},{source.Y:.0f},{source.Z:.0f})"
          f"  dst=({destination.X:.0f},{destination.Y:.0f},{destination.Z:.0f})")

    gripper_open()

    print("[MOVE] Going above box")
    robot.go_to(src_above, vel=97.0)
    add_trajectory_point(source.X, source.Y, box_label, color=(0,255,255))

    print(f"[MOVE] Lowering onto box: Z={source.Z:.0f}")
    robot.go_to(source, vel=97.0)

    gripper_close()
    winsound.Beep(1000, 200)
    say(f"{box_label} picked up.")

    print("[MOVE] Lifting box")
    robot.go_to(src_above, vel=97.0)

    print("[MOVE] Travelling to storage")
    robot.go_to(dst_above, vel=97.0)

    if _prev_pos[0]:
        update_travel(destination.X - _prev_pos[0][0],
                      destination.Y - _prev_pos[0][1])
    _prev_pos[0] = (destination.X, destination.Y)

    print(f"[MOVE] Lowering onto stack: Z={destination.Z:.0f}")
    robot.go_to(destination, vel=97.0)
    add_trajectory_point(destination.X, destination.Y,
                         storage_label, color=(255,100,0))

    gripper_open()
    winsound.Beep(600, 300)
    say("Box placed on stack.")

    print("[MOVE] Retreating")
    robot.go_to(dst_above, vel=97.0)


def pick_and_place_box(box: Box, storage_x, storage_y, stack_layer):
    pick_x = box.robot_x + PICK_TABLE_X_OFFSET + GRIPPER_X_OFFSET
    pick_y = box.robot_y + PICK_TABLE_Y_OFFSET + GRIPPER_Y_OFFSET
    pick_z = PICK_Z

    place_x = storage_x + STORAGE_TABLE_X_OFFSET
    place_y = storage_y + STORAGE_TABLE_Y_OFFSET
    place_z = PLACE_Z + stack_layer * BOX_HEIGHT

    source      = igus.Cart(pick_x, pick_y, pick_z,    *ORIENT)
    destination = igus.Cart(place_x, place_y, place_z,  *ORIENT)

    old_pos = {"X": round(pick_x,1), "Y": round(pick_y,1), "Z": pick_z}
    new_pos = {"X": place_x, "Y": place_y, "Z": place_z}

    box_label = f"Box {box.box_id}"
    ordinals  = ["first", "second", "third", "fourth"]
    ordinal   = ordinals[stack_layer] if stack_layer < len(ordinals) else str(stack_layer+1)

    print(f"\n[PICK]  {box_label}  "
          f"robot=({box.robot_x:.1f}, {box.robot_y:.1f})  layer={stack_layer}")

    publish_status(json.dumps({
        "box_id":  box.box_id,
        "pos_old": old_pos,
        "pos_new": new_pos,
    }))

    say(f"Picking the {ordinal} nearest box.")
    update_dash(current_box=box_label, status="RUNNING")
    set_display(active_id=box.box_id)

    _move(source, destination,
          box_label=box_label,
          storage_label=f"Stack {stack_layer+1}")

    clear_active()
    with _dash_lock:
        _dash["picked"] += 1

    say(f"{box_label} stacked successfully.")
    print(f"[PLACE] {box_label} placed at layer {stack_layer}.")


# ============================================================================
# MAIN SORTING ROUTINE
# ============================================================================

def run_sort(storage_x, storage_y):
    print("\n" + "=" * 60)
    print(f"SORTING STARTED  |  storage: X={storage_x} Y={storage_y} mm")
    print("=" * 60)

    clear_trajectory()
    _prev_pos[0] = None
    update_dash(
        status     = "DETECTING",
        detected   = 0,
        picked     = 0,
        total      = 0,
        current_box= "-",
        travel_m   = 0.0,
        start_time = time.time(),
    )

    try:
        storage_px = robot_to_pixel(storage_x, storage_y)
        set_display(storage_pixel=storage_px)
    except Exception:
        storage_px = None

    print("[SORT] Moving robot away from camera...")
    say("Scanning workspace. Please stand clear.")
    with _display_lock:
        _disp_boxes[0]  = []
        _disp_active[0] = None
    robot.go_to(FACE_AWAY_POSE, vel=97.0)
    time.sleep(2.0)

    print("[SORT] Detecting boxes...")
    frame = get_fresh_frame()
    boxes = detector.detect(frame)
    set_display(boxes=boxes)

    n = len(boxes)
    print(f"[SORT] {n} box(es) detected.")
    publish_status(f"{n} have been found!")
    update_dash(detected=n, total=n,
                status="RUNNING" if n > 0 else "IDLE")

    if n == 0:
        say("No boxes detected. Aborting.")
        winsound.Beep(300, 1000)
        return

    box_word = "box" if n == 1 else "boxes"
    say(f"{n} {box_word} detected. Starting pick and place sequence.")

    boxes_sorted = sorted(boxes, key=lambda b: b.distance_from_origin())
    print("[SORT] Order (nearest first):")
    for i, b in enumerate(boxes_sorted):
        print(f"  {i+1}. Box {b.box_id}  dist={b.distance_from_origin():.1f} mm")

    robot.go_to_L(vel=97.0)

    for layer, box in enumerate(boxes_sorted):
        pick_and_place_box(box, storage_x, storage_y, stack_layer=layer)
        frame = get_fresh_frame()
        boxes = detector.detect(frame)
        set_display(boxes=boxes)

    clear_active()
    with _display_lock:
        _disp_boxes[0] = []

    update_dash(status="COMPLETE", current_box="Done")
    publish_status("COMPLETE")
    print("\n[SORT] All boxes placed. COMPLETE.\n")

    say("Sorting complete. All boxes have been stacked successfully.")
    winsound.Beep(800, 200)
    winsound.Beep(800, 200)
    winsound.Beep(800, 200)

    robot.go_to_L(vel=97.0)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS SORTING SYSTEM - ready")
    print(f"Waiting for start message on:  {TOPIC_SORT}")
    print(f"Payload format:  {{\"X\": <mm>, \"Y\": <mm>}}")
    print("=" * 60 + "\n")

    say("System ready. Waiting for start command.")

    try:
        while True:
            if _sort_requested.is_set():
                _sort_requested.clear()
                run_sort(_storage_x, _storage_y)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[MAIN] Stopped by user.")
        say("System shutting down.")

    finally:
        _stop_event.set()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        robot.disconnect()
        detector.release()
        cv2.destroyAllWindows()
        print("[MAIN] Shutdown complete.")