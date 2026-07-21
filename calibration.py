"""1
What it does:
  1. Opens the camera and shows a live view.
  2. You click on 4 (or more) known points on the table in the image.
  3. For each clicked point you type the real robot X and Y coordinates
     (read from iRC while jogging — see instructions in the README).
  4. Press 's' to solve → it computes the pixel-to-robot transform matrix.
  5. Press 'q' to quit → it saves the matrix to  'transform.npy'
     so main.py can load it without repeating calibration.
"""

import cv2
import numpy as np

# ── camera source ──────────────────────────────────────────────────────────
# 0 = built-in webcam, 1 = first external USB camera.
# Change to 1 (or higher) if your top-down camera is external.
CAMERA_SOURCE = 0

TRANSFORM_FILE = "transform.npy"   # where the matrix is saved/loaded
WIN = "CALIBRATION"


class Calibration:
    def __init__(self, source=CAMERA_SOURCE):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera source {source!r}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.pixel_pts = []   # list of (u, v) pixel coordinates
        self.robot_pts = []   # list of (X, Y) robot coordinates in mm
        self.transform = None # 2x3 affine matrix once solved
        self.frame = None
        self.solved = False

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self._on_mouse)

    # ── mouse callback ────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.frame is None:
            return
        if self.solved:
            # In solved mode show what robot coordinate a pixel maps to
            rX, rY = self.pixel_to_robot(x, y)
            print(f"[INFO] pixel ({x},{y})  →  robot X={rX:.1f} mm  Y={rY:.1f} mm")
            return

        # Calibration mode: ask for the matching robot coordinate
        print(f"\nSelected pixel ({x}, {y}).")
        try:
            rx = float(input("  Robot X (mm): "))
            ry = float(input("  Robot Y (mm): "))
        except ValueError:
            print("  Not a number — point cancelled.")
            return
        self.pixel_pts.append((x, y))
        self.robot_pts.append((rx, ry))
        n = len(self.pixel_pts)
        print(f"  Point {n} added."
              + ("  Ready to solve — press 's'." if n >= 4 else ""))

    # ── solve ─────────────────────────────────────────────────────────────
    def solve(self):
        n = len(self.pixel_pts)
        if n < 4:
            print(f"[CALIBRATION] Need at least 4 points (have {n}).")
            return

        src = np.array(self.pixel_pts, dtype=np.float32)
        dst = np.array(self.robot_pts,  dtype=np.float32)

        # estimateAffinePartial2D fits a similarity transform (rotation +
        # uniform scale + translation) which is correct for a top-down
        # overhead camera with no perspective distortion.
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC)
        if M is None:
            print("[CALIBRATION] ERROR: could not fit a transform — "
                  "try adding more spread-out points.")
            return

        self.transform = M
        self.solved = True

        # Report residual error so you can judge calibration quality
        proj = (M @ np.hstack([src, np.ones((n, 1))]).T).T
        err = np.linalg.norm(proj - dst, axis=1)
        print(f"\n[CALIBRATION] Solved on {n} points | "
              f"mean error {err.mean():.2f} mm | max error {err.max():.2f} mm")
        print("[CALIBRATION] Calibration complete.  "
              "Click anywhere to test.  Press 'q' to save and quit.\n")

    def pixel_to_robot(self, u, v):
        """Convert one pixel (u,v) → robot (X,Y) in mm."""
        if self.transform is None:
            raise RuntimeError("Transform not solved yet.")
        X, Y = self.transform @ np.array([u, v, 1.0], dtype=np.float32)
        return float(X), float(Y)

    def save(self):
        if self.transform is not None:
            np.save(TRANSFORM_FILE, self.transform)
            print(f"[CALIBRATION] Transform saved to '{TRANSFORM_FILE}'.")
        else:
            print("[CALIBRATION] Nothing to save — transform was not solved.")

    def undo(self):
        if not self.solved and self.pixel_pts:
            self.pixel_pts.pop()
            self.robot_pts.pop()
            print(f"[CALIBRATION] Last point removed. {len(self.pixel_pts)} remaining.")

    # ── drawing ───────────────────────────────────────────────────────────
    def _draw(self, frame):
        for i, (u, v) in enumerate(self.pixel_pts):
            cv2.drawMarker(frame, (int(u), int(v)), (45, 45, 255),
                           cv2.MARKER_CROSS, 18, 2)
            cv2.circle(frame, (int(u), int(v)), 9, (45, 45, 255), 2)
            cv2.putText(frame, str(i + 1), (int(u) + 12, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 255), 1)

        if self.solved:
            label = "SOLVED — click to test pixel→robot | q=save & quit"
            color = (0, 220, 0)
        else:
            label = (f"CALIBRATE  {len(self.pixel_pts)} pts  "
                     f"| s=solve  u=undo  q=quit")
            color = (255, 255, 255)

        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 1, cv2.LINE_AA)

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self):
        print("=" * 60)
        print("CALIBRATION TOOL")
        print("=" * 60)
        print("Step 1: Jog the real robot to each corner of the table.")
        print("Step 2: Click that corner in the camera window.")
        print("Step 3: Type the robot X and Y you read off iRC.")
        print("Step 4: Repeat for 4+ corners, then press 's' to solve.")
        print("Step 5: Press 'q' to save and quit.\n")

        while True:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
            if self.frame is not None:
                shown = self.frame.copy()
                self._draw(shown)
                cv2.imshow(WIN, shown)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s') and not self.solved:
                self.solve()
            elif key == ord('u'):
                self.undo()
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.save()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Calibration().run()