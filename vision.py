import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import math

CAMERA_SOURCE  = 0
TRANSFORM_FILE = "transform.npy"

HSV_LOWER = np.array([18, 128, 108], dtype=np.uint8)
HSV_UPPER = np.array([28, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 80000
MAX_BOXES        = 3


@dataclass
class Box:
    box_id:      int
    robot_x:     float
    robot_y:     float
    pixel_bbox:  tuple = field(default_factory=tuple)  # kept for compatibility
    pixel_cx:    int   = 0
    pixel_cy:    int   = 0
    pixel_points: object = None   # rotated rectangle corner points (4x2 array)
    angle:       float = 0.0      # rotation angle of the box in degrees

    def distance_from_origin(self) -> float:
        return math.sqrt(self.robot_x ** 2 + self.robot_y ** 2)


class BoxDetector:
    def __init__(self, source: int = CAMERA_SOURCE,
                 transform_file: str = TRANSFORM_FILE):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera source {source!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            self.transform = np.load(transform_file)
            print(f"[Vision] Loaded transform from '{transform_file}'.")
        except FileNotFoundError:
            raise SystemExit(
                f"[Vision] ERROR: '{transform_file}' not found.\n"
                "Run calibration.py first to create it.")

    def pixel_to_robot(self, u: float, v: float):
        X, Y = self.transform @ np.array([u, v, 1.0], dtype=np.float32)
        return float(X), float(Y)

    def grab_frame(self):
        for _ in range(3):
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("[Vision] Could not read frame from camera.")
        return frame

    def detect(self, frame=None) -> List[Box]:
        if frame is None:
            frame = self.grab_frame()

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        valid = [c for c in contours
                 if MIN_CONTOUR_AREA < cv2.contourArea(c) < MAX_CONTOUR_AREA]
        valid.sort(key=cv2.contourArea, reverse=True)
        valid = valid[:MAX_BOXES]

        boxes = []
        for idx, cnt in enumerate(valid):
            # ── ROTATED bounding rectangle ──────────────────────────────────
            # minAreaRect returns (center, (width, height), angle)
            # It finds the smallest rectangle that fits the contour at any angle
            rect   = cv2.minAreaRect(cnt)
            center = rect[0]          # (cx, cy) in pixels
            angle  = rect[2]          # rotation angle in degrees

            # boxPoints gives the 4 corner pixels of the rotated rectangle
            pts = cv2.boxPoints(rect)
            pts = np.int32(pts)       # convert to integer pixel coordinates

            cx = int(center[0])
            cy = int(center[1])

            # axis-aligned bbox still stored for label positioning
            x, y, w, h = cv2.boundingRect(cnt)

            robot_x, robot_y = self.pixel_to_robot(cx, cy)

            boxes.append(Box(
                box_id       = idx,
                robot_x      = robot_x,
                robot_y      = robot_y,
                pixel_bbox   = (x, y, w, h),
                pixel_cx     = cx,
                pixel_cy     = cy,
                pixel_points = pts,    # 4 corner points of rotated rectangle
                angle        = angle,
            ))

        return boxes

    def draw(self, frame: np.ndarray, boxes: List[Box],
             active_id: Optional[int] = None,
             storage_pixel: Optional[tuple] = None) -> np.ndarray:
        out = frame.copy()

        for box in boxes:
            box_color    = (0, 255, 0) if box.box_id == active_id else (0, 0, 255)
            marker_color = (0, 255, 0) if box.box_id == active_id else (0, 255, 255)

            # ── draw ROTATED bounding rectangle ─────────────────────────────
            # drawContours draws the 4 connected corner points as a polygon
            if box.pixel_points is not None:
                cv2.drawContours(out, [box.pixel_points], 0, box_color, 2)
            else:
                # fallback to axis-aligned if no rotated points
                x, y, w, h = box.pixel_bbox
                cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 2)

            # centre cross marker
            cv2.drawMarker(out, (box.pixel_cx, box.pixel_cy),
                           marker_color, cv2.MARKER_CROSS, 16, 2)

            # label — position above the axis-aligned bbox top edge
            x, y, w, h = box.pixel_bbox
            label = f"Box {box.box_id}  ({box.robot_x:.0f},{box.robot_y:.0f})  {box.angle:.1f}deg"
            cv2.putText(out, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

        # blue storage spot
        if storage_pixel is not None:
            sx, sy = storage_pixel
            cv2.rectangle(out, (sx - 40, sy - 40), (sx + 40, sy + 40),
                          (255, 80, 0), 2)
            cv2.putText(out, "STORAGE", (sx - 35, sy - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)

        # status
        status = f"Detected: {len(boxes)} box(es)"
        cv2.putText(out, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    WIN_CAM  = "Camera (original)"
    WIN_MASK = "HSV Mask - tune until only boxes are white"

    try:
        detector = BoxDetector()
    except SystemExit as e:
        print(e)
        print("[Tuning] Continuing without calibration.")
        detector = None
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = detector.cap

    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H low",  WIN_MASK, int(HSV_LOWER[0]), 180, lambda v: None)
    cv2.createTrackbar("S low",  WIN_MASK, int(HSV_LOWER[1]), 255, lambda v: None)
    cv2.createTrackbar("V low",  WIN_MASK, int(HSV_LOWER[2]), 255, lambda v: None)
    cv2.createTrackbar("H high", WIN_MASK, int(HSV_UPPER[0]), 180, lambda v: None)
    cv2.createTrackbar("S high", WIN_MASK, int(HSV_UPPER[1]), 255, lambda v: None)
    cv2.createTrackbar("V high", WIN_MASK, int(HSV_UPPER[2]), 255, lambda v: None)

    print("Adjust sliders until only the boxes appear white.")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        lo = np.array([cv2.getTrackbarPos("H low",  WIN_MASK),
                       cv2.getTrackbarPos("S low",  WIN_MASK),
                       cv2.getTrackbarPos("V low",  WIN_MASK)], dtype=np.uint8)
        hi = np.array([cv2.getTrackbarPos("H high", WIN_MASK),
                       cv2.getTrackbarPos("S high", WIN_MASK),
                       cv2.getTrackbarPos("V high", WIN_MASK)], dtype=np.uint8)

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow(WIN_CAM,  frame)
        cv2.imshow(WIN_MASK, mask)

        print(f"\rHSV_LOWER=({lo[0]},{lo[1]},{lo[2]})  "
              f"HSV_UPPER=({hi[0]},{hi[1]},{hi[2]})    ", end="")

        if cv2.waitKey(20) & 0xFF in (ord('q'), 27):
            break
        if cv2.getWindowProperty(WIN_MASK, cv2.WND_PROP_VISIBLE) < 1:
            break

    print()
    cap.release()
    cv2.destroyAllWindows()