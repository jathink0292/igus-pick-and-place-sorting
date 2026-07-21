# igus-pick-and-place-sorting
# Autonomous Pick-and-Place Sorting System — igus ReBeL 6-DOF

A vision-guided robotic cell that autonomously detects, localizes, and sorts parts
by colour using an **igus ReBeL 6-DOF** arm. A fixed overhead camera identifies the
target boxes, a calibrated transform converts image coordinates into real-world robot
coordinates, and the arm performs vacuum pick-and-place — stacking each box at a
requested storage location — with no manual intervention.

A start command and the storage location are received over **MQTT**, and the system
publishes live status back to the same broker while showing an on-screen dashboard.

Built as the final robotics project for my M.Eng. (Industry 4.0 — Automation, Robotics
& 3D Manufacturing) at SRH University Berlin.

## Demo
https://github.com/user-attachments/assets/ed28f50a-0f45-421e-bff7-55a7470129ac

https://github.com/user-attachments/assets/c4cd730b-b74e-4e7c-a6a7-42a1a13224ac


## How it works

1. **Perception** (`vision.py`) — OpenCV HSV colour segmentation detects boxes and
   returns each box's centre and orientation in image (pixel) coordinates.
2. **Calibration** (`calibration.py`) — an affine transform maps camera pixels to robot
   coordinates (saved to `transform.npy` and reused at run time).
3. **Control** (`main.py`) — waits for an MQTT start command, converts box positions to
   robot coordinates, and runs the pick → travel → place → stack sequence, driving the
   vacuum gripper and reporting status over MQTT.

## Repository structure

| File | Purpose | Author |
|------|---------|--------|
| `vision.py` | Camera capture and HSV-based box detection | Jathin Kandala |
| `calibration.py` | Camera-to-robot coordinate calibration (affine transform) | Jathin Kandala |
| `main.py` | Control loop, MQTT, and pick-and-place motion sequence | Jathin Kandala |
| `igus.py` | Low-level robot communication interface (CRI protocol) | Provided by course instructor (see header in file) |

## Tech stack

- **Language:** Python
- **Vision:** OpenCV (HSV colour detection)
- **Robot:** igus ReBeL 6-DOF, CRI protocol over TCP socket
- **Messaging:** MQTT (paho-mqtt)
- **Numerics:** NumPy

## Setup

```bash
pip install -r requirements.txt
```

Set your robot IP/port and MQTT broker at the top of `main.py` before running. Then:

```bash
python calibration.py   # one-time: create transform.npy
python main.py          # run the sorter (waits for an MQTT start command)
```

> Developed and run on Windows — `main.py` uses `winsound` and PowerShell speech for
> audio feedback.

## Credits

`igus.py` is the robot communication interface provided by the course instructor at
SRH University Berlin (attribution retained in the file header). All other files —
vision, calibration, control logic, and the pick-and-place sequence — are my own work.

## Author

**Jathin Kandala** — M.Eng. Industry 4.0, SRH University Berlin
