import cv2
import time
import math
import csv
import os
import threading
import queue
import numpy as np
from face_recognition.recognize import get_person_name, init_face_recognition
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
try:
    mp_styles = mp.solutions.drawing_styles
except AttributeError:
    mp_styles = None  # fallback if styles not available

# =========================
# Feature toggles & config
# =========================
TARGET_SECONDS       = 20          # WHO minimum
MOTION_THRESH        = 2.5         # avg px/frame to count as "washing"
REQUIRED_HANDS       = 2           # expect two hands
CAM_INDEX            = 0
DRAW_LANDMARKS       = True

ENABLE_VOICE         = True
VOICE_RATE_WPM       = 180
ENABLE_LOGGING       = True
LOG_FILE             = "handwash_log.csv"
ENABLE_GUIDANCE      = True

GUIDE_STEPS = [
    "Palms to palms",
    "Backs of hands",
    "Between fingers",
    "Around thumbs",
    "Nails & wrists",
]

# =========================
# Voice helper
# =========================
class Speaker:
    def __init__(self, enable=True, rate=180):
        self.enable = enable
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.thread = None
        self.engine = None
        if enable:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', rate)
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            except Exception as e:
                print(f"[Voice] Disabled (init error: {e})")
                self.enable = False

    def _run(self):
        import pyttsx3
        while not self._stop.is_set():
            try:
                msg = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.engine.say(msg)
                self.engine.runAndWait()
            except:
                pass

    def say(self, msg: str):
        if self.enable:
            self.q.put(msg)

    def stop(self):
        self._stop.set()

# =========================
# Helpers
# =========================
def landmark_px_list(landmarks, w, h):
    return [(int(lm.x*w), int(lm.y*h)) for lm in landmarks.landmark]

def mean_l2_dist(a, b):
    if a is None or b is None or len(a) != len(b):
        return 0.0
    return sum(math.hypot(x1-x2, y1-y2) for (x1,y1),(x2,y2) in zip(a,b))/len(a)

def ratio_points_in_roi(points, roi):
    if points is None or roi is None:
        return 0.0
    x, y, w, h = roi
    if w <=0 or h <=0:
        return 0.0
    inside = sum(1 for px,py in points if x<=px<=x+w and y<=py<=y+h)
    return inside/len(points) if points else 0.0

def draw_progress_bar(img, progress, msg, color=(0,200,0)):
    h, w = img.shape[:2]
    bar_h = 26
    pad = 10
    x1, y1 = pad, h-bar_h-pad
    x2, y2 = w-pad, h-pad
    cv2.rectangle(img, (x1,y1),(x2,y2),(40,40,40),-1)
    fill_w = int((x2-x1)*np.clip(progress,0,1))
    cv2.rectangle(img,(x1,y1),(x1+fill_w,y2),color,-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),(200,200,200),1)
    cv2.putText(img,msg,(x1+8,y1-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(240,240,240),2,cv2.LINE_AA)

def pick_roi_once(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    cv2.putText(frame,"Draw the WASHING ZONE, press ENTER. Press C to cancel.",(12,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imshow("Select Washing Zone", frame)
    cv2.waitKey(300)
    roi = cv2.selectROI("Select Washing Zone", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Washing Zone")
    x,y,w,h = roi
    return (int(x),int(y),int(w),int(h)) if w>0 and h>0 else None

def ensure_log_header(path):
    import os
    if not os.path.exists(path):
        with open(path,"w",newline="",encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["timestamp","duration_sec","completed","hands_required",
                             "motion_thresh","roi_x","roi_y","roi_w","roi_h","person_name"])

def log_session(path,duration,completed,roi,person_name="Unknown"):
    ensure_log_header(path)
    import csv
    x=[None,None,None,None]
    if roi:
        x=[roi[0],roi[1],roi[2],roi[3]]
    with open(path,"a",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), round(duration,2), int(completed),
                         REQUIRED_HANDS,MOTION_THRESH,*x,person_name])

def draw_guidance_panel(img, steps, completed_steps, total_steps):
    """
    Draws a guidance panel showing step progress.

    Args:
        img: Frame to draw on
        steps: List of step labels
        completed_steps: Number of steps completed so far (int)
        total_steps: Total number of steps (len(steps))
    """
    if not steps:
        return

    h, w = img.shape[:2]
    panel_w = 320
    x0, y0 = w - panel_w - 10, 10
    x1, y1 = w - 10, 10 + 24 + len(steps) * 28 + 10

    # Background
    cv2.rectangle(img, (x0, y0), (x1, y1), (25, 25, 25), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
    cv2.putText(img, "Guided Steps", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw each step
    for i, label in enumerate(steps):
        y = y0 + 24 + (i + 1) * 28

        # ✅ mark step as completed if its index < completed_steps
        done = i < completed_steps
        # ✅ current active step is the next one to complete
        active = (i == completed_steps and completed_steps < total_steps)

        if done:
            box_col = (0, 180, 0)  # green
        elif active:
            box_col = (60, 180, 255)  # orange
        else:
            box_col = (120, 120, 120)  # grey

        # Checkbox
        cv2.rectangle(img, (x0 + 10, y - 18), (x0 + 32, y + 2), box_col, 2)
        if done:
            # Tick mark
            cv2.line(img, (x0 + 12, y - 8), (x0 + 20, y), box_col, 2)
            cv2.line(img, (x0 + 20, y), (x0 + 32, y - 14), box_col, 2)

        # Step text
        cv2.putText(img, label, (x0 + 40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2)

def detect_step(hands_pts, step_index):
    """
    Rule-based gesture detector for WHO handwash steps.
    Very simplified heuristics for demo purposes.
    """
    if len(hands_pts) < 2:
        return False

    left, right = hands_pts[0], hands_pts[1]

    def avg_point(pts, idx):
        return np.mean([p[idx] for p in pts])

    # Example rules (tuned heuristics):
    if step_index == 0:  # Palms to palms
        dist = math.hypot(avg_point(left,0)-avg_point(right,0),
                          avg_point(left,1)-avg_point(right,1))
        return dist < 200  # hands close together

    elif step_index == 1:  # Backs of hands
        return left[0][0] < right[0][0] or right[0][0] < left[0][0]

    elif step_index == 2:  # Between fingers
        return abs(avg_point(left,1)-avg_point(right,1)) > 50

    elif step_index == 3:  # Around thumbs
        return abs(left[4][0]-right[4][0]) < 50  # thumb tip close

    elif step_index == 4:  # Nails & wrists
        return left[8][1] > left[0][1] or right[8][1] > right[0][1]

    return False

# =========================
# Main
# =========================
def main():

    from face_recognition.recognize import get_person_name, init_face_recognition, detector, recognizer, face_db

    init_face_recognition(
        detector_weights="face_recognition/weights/det_10g.onnx",
        recognizer_weights="face_recognition/weights/w600k_mbf.onnx",
        db_path="face_recognition/database/face_database"
    )

    step_start_time = None

    roi_coords = None  
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    roi = pick_roi_once(cap)
    last_frame_time = time.time()
    accumulated_sec = 0.0
    washing_active = False
    session_logged = False
    hands_absent_timer = 0.0
    ABSENCE_LIMIT = 25.0  # seconds
    step_index = 0  # current step user is on
    step_start_time = None
    STEP_MIN_DURATION = TARGET_SECONDS / len(GUIDE_STEPS)  # minimum seconds per step

    prev_hand1=None
    prev_hand2=None
    ema_motion=0.0
    alpha=0.3

    current_person = None
    session_person = None

    speaker = Speaker(ENABLE_VOICE,VOICE_RATE_WPM)
    spoke_start=spoke_half=spoke_complete=False
    spoke_countdown_for=set()

    last_detected_person = None

    with mp_hands.Hands(model_complexity=0,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands_detector:
        while True:
            ok,frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame,1)
            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result = hands_detector.process(rgb)

            now = time.time()
            dt = now-last_frame_time
            last_frame_time = now

            hands_pts=[]
            if result.multi_hand_landmarks:
                for hls in result.multi_hand_landmarks:
                    pts=landmark_px_list(hls,w,h)
                    hands_pts.append(pts)
                    if DRAW_LANDMARKS and mp_styles:
                        mp_drawing.draw_landmarks(frame,hls,mp_hands.HAND_CONNECTIONS,
                                                  mp_styles.get_default_hand_landmarks_style(),
                                                  mp_styles.get_default_hand_connections_style())

            if roi:
                x,y,rw,rh = roi
                cv2.rectangle(frame,(x,y),(x+rw,y+rh),(255,200,0),2)
                cv2.putText(frame,"Washing Zone",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,200,0),2)
            else:
                cv2.putText(frame,"No washing zone set (press Z to set).",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            in_zone_counts=[]
            for pts in hands_pts:
                in_zone_counts.append(ratio_points_in_roi(pts,roi)>=0.6 if roi else True)

            motion_vals=[]
            if len(hands_pts)>=1:
                motion_vals.append(mean_l2_dist(hands_pts[0],prev_hand1))
            if len(hands_pts)>=2:
                motion_vals.append(mean_l2_dist(hands_pts[1],prev_hand2))

            prev_hand1 = hands_pts[0] if len(hands_pts)>=1 else None
            prev_hand2 = hands_pts[1] if len(hands_pts)>=2 else None

            motion = np.mean(motion_vals) if motion_vals else 0.0
            ema_motion = alpha*motion + (1-alpha)*ema_motion

            hands_present = len(hands_pts) >= REQUIRED_HANDS
            hands_in_zone = sum(in_zone_counts) >= REQUIRED_HANDS if roi else hands_present
            moving_enough = ema_motion >= MOTION_THRESH

            # --------- STEP PROGRESSION ----------
            if hands_present and hands_in_zone:
                hands_absent_timer = 0.0  # reset absence timer

                if moving_enough and washing_active:
                    if step_index < len(GUIDE_STEPS):
                        if detect_step(hands_pts, step_index):
                            if step_start_time is None:
                                step_start_time = now
                        if step_start_time is not None and now - step_start_time >= STEP_MIN_DURATION:
                                step_index += 1
                                step_start_time = now
                                print(f"[INFO] Step {step_index}/{len(GUIDE_STEPS)} completed")
                    else:
                        step_start_time = None  # reset if wrong gesture

            else:
                # hands not in zone
                if washing_active:
                    hands_absent_timer += dt
                    if hands_absent_timer >= ABSENCE_LIMIT:
                        safe_roi = roi if roi is not None else (0,0,0,0)
                        safe_person = session_person if session_person else "Unknown"
                        log_session(LOG_FILE, accumulated_sec, False, safe_roi, safe_person)
                        print(f"[INFO] Session ended due to inactivity ({ABSENCE_LIMIT}s)")
                        accumulated_sec = 0.0
                        washing_active = False
                        session_logged = True
                        hands_absent_timer = 0.0
                        step_index = 0
                        step_start_time = None
 

            # --------- SESSION STATE ----------
            # Only ask for name if session_person is None (new person)

            if session_person is None:
                person = get_person_name(frame)

                if person is not None and person != "Unknown":
                    session_person = person

            current_person = session_person

            if session_person != last_detected_person:
                accumulated_sec = 0.0
                washing_active = False
                session_logged = False
                # update last_detected_person to the new person
                last_detected_person = session_person
                spoke_start = spoke_half = spoke_complete = False
                spoke_countdown_for.clear()
                step_index = 0

            if hands_present and hands_in_zone and moving_enough:
                if not washing_active:
                    hands_absent_timer = 0.0
                    washing_active = True
                    accumulated_sec = 0.0
                    spoke_start = spoke_half = spoke_complete = False
                accumulated_sec += dt
            else:
                if washing_active:
                    hands_absent_timer += dt
                    if hands_absent_timer >= ABSENCE_LIMIT:
                        # session ends automatically
                        safe_roi = roi if roi is not None else (0,0,0,0)
                        safe_person = session_person if session_person else "Unknown"
                        log_session(LOG_FILE, accumulated_sec, False, safe_roi, safe_person)
                        print(f"[INFO] Session ended due to inactivity ({ABSENCE_LIMIT}s)")
                        accumulated_sec = 0.0
                        washing_active = False
                        session_logged = True
                        hands_absent_timer = 0.0
                        spoke_start = spoke_half = spoke_complete = False


            if washing_active:
                draw_guidance_panel(frame, GUIDE_STEPS, step_index, len(GUIDE_STEPS))
                if accumulated_sec >= TARGET_SECONDS and not session_logged:
                    safe_roi = roi if roi is not None else (0, 0, 0, 0)
                    log_session(LOG_FILE, accumulated_sec, True, safe_roi, session_person)
                    session_logged = True
            else:
                cv2.putText(frame, "Wash your hands!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"Detected: {current_person}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # --------- VOICE PROMPTS ----------
            if ENABLE_VOICE and washing_active:
                if not spoke_start and accumulated_sec>0.2:
                    speaker.say("Start washing")
                    if ENABLE_GUIDANCE and GUIDE_STEPS:
                        speaker.say("Step one. Palms to palms.")
                    spoke_start=True
                half = TARGET_SECONDS/2
                if accumulated_sec>=half and not spoke_half and accumulated_sec<TARGET_SECONDS:
                    speaker.say("Ten seconds left")
                    spoke_half=True
                remaining=int(max(0,math.ceil(TARGET_SECONDS-accumulated_sec)))
                if 1<=remaining<=5 and remaining not in spoke_countdown_for:
                    speaker.say(str(remaining))
                    spoke_countdown_for.add(remaining)
                if accumulated_sec>=TARGET_SECONDS and not spoke_complete:
                    speaker.say("Handwash complete. Great job.")
                    spoke_complete=True
                    if ENABLE_LOGGING and not session_logged:
                        safe_roi = roi if roi is not None else (0, 0, 0, 0)
                        safe_person = session_person if session_person else "Unknown"

                        try:
                            log_session(LOG_FILE, accumulated_sec, True, safe_roi, safe_person)
                        except Exception as e:
                            print(f"[ERROR] Could not log session: {e}")
                        session_logged=True

            # --------- STATUS BAR ----------
            progress = accumulated_sec/TARGET_SECONDS
            if accumulated_sec>=TARGET_SECONDS:
                status=f"Complete ✅  ({int(accumulated_sec)}s)"
                color=(0,200,0)
            else:
                if washing_active:
                    status=f"Washing…  {int(accumulated_sec)}/{TARGET_SECONDS}s"
                    color=(60,180,255)
                else:
                    status=f"Incomplete ❌  {int(accumulated_sec)}/{TARGET_SECONDS}s" if accumulated_sec>0 else "Show both hands in zone!"
                    color=(0,120,255)
            draw_progress_bar(frame,min(progress,1.0),status,color)
            cv2.imshow("Handwash Monitor", frame)


            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            elif key==ord('r'):
                if ENABLE_LOGGING and washing_active and not session_logged:
                    log_session(LOG_FILE,accumulated_sec,accumulated_sec>=TARGET_SECONDS,roi,session_person)
                    session_logged=True
                accumulated_sec=0.0
                washing_active=False
                session_logged=False
                session_person=None
                spoke_start=spoke_half=spoke_complete=False
                step_start_time = None
                spoke_countdown_for.clear()
            elif key==ord('z'):
                roi=pick_roi_once(cap)

    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()

if __name__=="__main__":
    main()