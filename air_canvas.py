import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import math
from collections import deque, Counter

# =========================================================
# Setup
# =========================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

CANVAS_W, CANVAS_H = 640, 480

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # red, green, blue, yellow
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW"]
BRUSH_SIZE = 8
ERASER_SIZE = 30
PALETTE_H = 50

CLEAR_HOLD_SECONDS = 0.7   # hold open palm this long to clear
SMOOTHING = 0.45           # 0..1, higher = more responsive, lower = smoother
MARGIN = 0.10              # ignore outer 10% of camera view
STABLE_FRAMES = 4          # frames used for gesture voting
SHOW_DEBUG = True

# Move / grab settings
GRAB_RADIUS = 45           # pixels: pinch this close to a stroke to grab it
PINCH_ON = 0.28            # thumb-index distance / hand size to START a pinch
PINCH_OFF = 0.42           # ...and to RELEASE it (hysteresis avoids flicker)


# =========================================================
# Camera
# =========================================================
def open_camera():
    backends = [("DSHOW", getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
                ("MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
                ("DEFAULT", cv2.CAP_ANY)]
    for index in range(3):
        for name, backend in backends:
            print(f"Trying camera index {index} with {name} ...")
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)
            ok = False
            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
            if ok:
                print(f"OK: Camera {index} ({name}) opened.")
                return cap
            cap.release()
    print("ERROR: No working camera found.")
    sys.exit(1)


# =========================================================
# Hand helpers
# =========================================================
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(lm):
    """[thumb, index, middle, ring, pinky] as 1/0, rotation independent."""
    wrist = lm[0]
    flags = [1 if dist(lm[4], lm[17]) > dist(lm[3], lm[17]) else 0]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        flags.append(1 if dist(lm[tip], wrist) > dist(lm[pip], wrist) else 0)
    return flags


def classify(fingers):
    """Thumb ignored on purpose (least reliable finger)."""
    _, idx, mid, ring, pinky = fingers
    if idx and mid and ring and pinky:
        return "PALM"
    if not (idx or mid or ring or pinky):
        return "FIST"
    if idx and mid and not ring and not pinky:
        return "ERASE"
    if idx and not mid and not ring and not pinky:
        return "DRAW"
    return "NONE"


def pinch_ratio(lm):
    """Thumb-tip to index-tip distance relative to hand size."""
    size = dist(lm[0], lm[9])
    return dist(lm[4], lm[8]) / max(size, 1e-6)


def to_canvas(x, y):
    cx = float(np.interp(x, [MARGIN, 1 - MARGIN], [0, CANVAS_W - 1]))
    cy = float(np.interp(y, [MARGIN, 1 - MARGIN], [0, CANVAS_H - 1]))
    return cx, cy


# =========================================================
# Stroke (object) helpers
# A stroke = {"color": (b, g, r), "pts": [[x, y], ...]}
# =========================================================
def point_to_polyline_dist(p, pts):
    P = np.array(pts, dtype=np.float32)
    p = np.array(p, dtype=np.float32)
    if len(P) == 1:
        return float(np.linalg.norm(P[0] - p))
    A, B = P[:-1], P[1:]
    AB = B - A
    denom = np.maximum((AB ** 2).sum(axis=1), 1e-6)
    t = np.clip(((p - A) * AB).sum(axis=1) / denom, 0, 1)
    proj = A + t[:, None] * AB
    return float(np.linalg.norm(proj - p, axis=1).min())


def points_to_segment_dist(P, a, b):
    ab = b - a
    denom = float(ab.dot(ab))
    if denom < 1e-6:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip(((P - a) @ ab) / denom, 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(P - proj, axis=1)


def find_nearest_stroke(strokes, p):
    best, best_d = None, GRAB_RADIUS
    for s in strokes:
        d = point_to_polyline_dist(p, s["pts"])
        if d < best_d:
            best, best_d = s, d
    return best


def erase_along(strokes, a, b, radius):
    """Remove stroke points near segment a-b; split strokes where cut."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    out = []
    for s in strokes:
        P = np.array(s["pts"], dtype=np.float32)
        keep = points_to_segment_dist(P, a, b) > radius
        if keep.all():
            out.append(s)
            continue
        run = []
        for pt, k in zip(s["pts"], keep):
            if k:
                run.append(pt)
            elif run:
                out.append({"color": s["color"], "pts": run})
                run = []
        if run:
            out.append({"color": s["color"], "pts": run})
    return out


def translate(stroke, dx, dy):
    for pt in stroke["pts"]:
        pt[0] += dx
        pt[1] += dy


def draw_stroke(canvas, pts, color, thick):
    arr = np.array(pts, dtype=np.float32).round().astype(np.int32)
    if len(arr) == 1:
        cv2.circle(canvas, (int(arr[0][0]), int(arr[0][1])), max(thick // 2, 1), color, cv2.FILLED)
    else:
        cv2.polylines(canvas, [arr.reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)


def render(strokes, highlight=None):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    for s in strokes:
        if s is highlight:
            draw_stroke(canvas, s["pts"], (255, 255, 255), BRUSH_SIZE + 6)
        draw_stroke(canvas, s["pts"], s["color"], BRUSH_SIZE)
    return canvas


# =========================================================
# UI
# =========================================================
def draw_color_palette(img, current_color):
    zone_w = CANVAS_W // (len(COLORS) + 1)
    for i, color in enumerate(COLORS):
        x1, x2 = i * zone_w, (i + 1) * zone_w
        cv2.rectangle(img, (x1, 0), (x2, PALETTE_H), color, cv2.FILLED)
        if i == current_color:
            cv2.rectangle(img, (x1, 0), (x2, PALETTE_H), (255, 255, 255), 3)
        text_col = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
        cv2.putText(img, COLOR_NAMES[i], (x1 + 10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2)
    x1 = len(COLORS) * zone_w
    cv2.rectangle(img, (x1, 0), (CANVAS_W, PALETTE_H), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, "ERASER", (x1 + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# =========================================================
# Main
# =========================================================
def main():
    cap = open_camera()
    zone_w = CANVAS_W // (len(COLORS) + 1)

    strokes = []               # all drawn objects
    current_stroke = None      # stroke being drawn right now
    prev_point = None          # for eraser line
    smooth = None
    current_color = 0
    palm_start = None
    history = deque(maxlen=STABLE_FRAMES)
    prev_time = 0
    failed_frames = 0

    # grab state
    grabbing = False
    pinch_count = 0
    grab_target = None         # a stroke dict, or "ALL"
    grab_last = None

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                success, img = cap.read()
                if not success or img is None:
                    failed_frames += 1
                    if failed_frames > 30:
                        print("ERROR: Camera stopped delivering frames.")
                        break
                    time.sleep(0.03)
                    continue
                failed_frames = 0

                if img.shape[1] != CANVAS_W or img.shape[0] != CANVAS_H:
                    img = cv2.resize(img, (CANVAS_W, CANVAS_H))

                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = hands.process(img_rgb)

                draw_color_palette(img, current_color)
                status = "Show your hand..."
                gesture = "NONE"
                fingers = [0, 0, 0, 0, 0]

                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    lm = hand_lm.landmark

                    mp_draw.draw_landmarks(
                        img, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

                    fingers = fingers_up(lm)
                    history.append(classify(fingers))
                    gesture, votes = Counter(history).most_common(1)[0]
                    if votes < max(2, len(history) // 2 + 1):
                        gesture = "NONE"

                    # ---------- Pinch detection (with hysteresis) ----------
                    ratio = pinch_ratio(lm)
                    was_grabbing = grabbing
                    if grabbing:
                        if ratio > PINCH_OFF:
                            grabbing = False
                            pinch_count = 0
                    else:
                        pinch_count = pinch_count + 1 if ratio < PINCH_ON else 0
                        grabbing = pinch_count >= 2
                    grab_started = grabbing and not was_grabbing

                    # ---------- Pointer position ----------
                    if grabbing:
                        # midpoint between thumb tip and index tip
                        raw = to_canvas((lm[4].x + lm[8].x) / 2, (lm[4].y + lm[8].y) / 2)
                    else:
                        raw = to_canvas(lm[8].x, lm[8].y)

                    if smooth is None or grabbing != was_grabbing:
                        smooth = raw   # snap on mode change to avoid a jump
                    else:
                        smooth = (smooth[0] + (raw[0] - smooth[0]) * SMOOTHING,
                                  smooth[1] + (raw[1] - smooth[1]) * SMOOTHING)
                    ix, iy = int(smooth[0]), int(smooth[1])

                    # ---------- GRAB / MOVE ----------
                    if grabbing:
                        gesture = "GRAB"
                        palm_start = None
                        prev_point = None

                        if grab_started:
                            # drop tiny accidental stroke made while closing the pinch
                            if current_stroke is not None and len(current_stroke["pts"]) <= 4:
                                strokes = [s for s in strokes if s is not current_stroke]
                            current_stroke = None

                            target = find_nearest_stroke(strokes, (ix, iy))
                            if target is not None:
                                grab_target = target
                            elif strokes:
                                grab_target = "ALL"
                            else:
                                grab_target = None
                            grab_last = (ix, iy)

                        if grab_target is not None and grab_last is not None:
                            dx, dy = ix - grab_last[0], iy - grab_last[1]
                            if grab_target == "ALL":
                                for s in strokes:
                                    translate(s, dx, dy)
                                status = "Moving whole drawing"
                            else:
                                translate(grab_target, dx, dy)
                                status = "Moving stroke"
                            grab_last = (ix, iy)
                        else:
                            status = "Nothing to move"

                        cv2.circle(img, (ix, iy), 12, (0, 255, 0), 3)

                    else:
                        grab_target = None
                        grab_last = None
                        cv2.circle(img, (ix, iy), 8, (255, 255, 255), cv2.FILLED)

                        # ---------- Gesture logic ----------
                        if gesture == "PALM":
                            prev_point = None
                            current_stroke = None
                            if palm_start is None:
                                palm_start = time.time()
                            held = time.time() - palm_start
                            if held >= CLEAR_HOLD_SECONDS:
                                strokes.clear()
                                palm_start = None
                                status = "Canvas Cleared!"
                            else:
                                status = f"Hold to clear... {held:.1f}s"
                        else:
                            palm_start = None

                            if gesture == "FIST":
                                prev_point = None
                                current_stroke = None
                                status = "Pen Up (hovering)"

                            elif gesture == "ERASE":
                                current_stroke = None
                                status = "ERASER"
                                if prev_point:
                                    strokes = erase_along(strokes, prev_point, (ix, iy),
                                                          ERASER_SIZE // 2 + 4)
                                else:
                                    strokes = erase_along(strokes, (ix, iy), (ix, iy),
                                                          ERASER_SIZE // 2 + 4)
                                prev_point = (ix, iy)
                                cv2.circle(img, (ix, iy), ERASER_SIZE // 2, (200, 200, 200), 2)

                            elif gesture == "DRAW":
                                if iy < PALETTE_H + 5:
                                    zone = ix // zone_w
                                    if zone < len(COLORS):
                                        current_color = zone
                                        status = f"Color: {COLOR_NAMES[current_color]}"
                                    else:
                                        status = "Use 2 fingers to erase"
                                    prev_point = None
                                    current_stroke = None
                                else:
                                    if current_stroke is None:
                                        current_stroke = {"color": COLORS[current_color], "pts": []}
                                        strokes.append(current_stroke)
                                    current_stroke["pts"].append([float(ix), float(iy)])
                                    prev_point = (ix, iy)
                                    status = f"Drawing ({COLOR_NAMES[current_color]})"
                            else:
                                prev_point = None
                                current_stroke = None
                                status = "Unclear gesture"
                else:
                    # hand lost
                    prev_point = None
                    current_stroke = None
                    smooth = None
                    palm_start = None
                    grabbing = False
                    pinch_count = 0
                    grab_target = None
                    grab_last = None
                    history.clear()

                # ---------- Render strokes onto camera frame ----------
                highlight = grab_target if isinstance(grab_target, dict) else None
                canvas = render(strokes, highlight)
                mask = canvas.any(axis=2)
                img[mask] = canvas[mask]

                # ---------- Overlay ----------
                cv2.putText(img, status, (10, CANVAS_H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, "Index=Draw | 2 fingers=Erase | Pinch=Move (empty area = move all)",
                            (10, CANVAS_H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(img, "Fist=Pen up | Palm(hold)=Clear | ESC=Exit",
                            (10, CANVAS_H - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                if SHOW_DEBUG:
                    cv2.putText(img, f"{gesture}  T{fingers[0]} I{fingers[1]} M{fingers[2]} R{fingers[3]} P{fingers[4]}",
                                (10, PALETTE_H + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time else 0
                prev_time = curr_time
                cv2.putText(img, f"FPS: {int(fps)}", (CANVAS_W - 130, CANVAS_H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Air Canvas", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()