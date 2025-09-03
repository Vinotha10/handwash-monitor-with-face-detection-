import cv2
import os
from face_recognition.main import SCRFD, ArcFace, build_face_database
import time

# Global handles
detector = None
recognizer = None
face_db = None

# Path to store faces
faces_dir = os.path.join(os.path.dirname(__file__), "assets", "faces")
os.makedirs(faces_dir, exist_ok=True)

def init_face_recognition(detector_weights, recognizer_weights, db_path):
    """
    Initialize models and face database.
    """
    global detector, recognizer, face_db

    detector = SCRFD(detector_weights, input_size=(640, 640), conf_thres=0.5)
    recognizer = ArcFace(recognizer_weights)

    # Build database from existing faces
    args = type('Args', (), {
        'faces_dir': faces_dir,
        'db_path': db_path,
        'update_db': False,
        'similarity_thresh': 0.5,
        'max_num': 1,
        'det_weight': detector_weights,
        'rec_weight': recognizer_weights,
        'confidence_thresh': 0.5
    })()
    face_db = build_face_database(detector, recognizer, args)

def get_person_name(frame, threshold=0.5):
    """
    Hybrid face recognition:
    1. Search database for known faces.
    2. If known → return name.
    3. If unknown → ask user input, save face image, recompute embedding from saved image,
       add to DB, save DB to disk, and return new name.
    """
    global detector, recognizer, face_db

    # 1) run detector on the frame
    bboxes, kpss = detector.detect(frame, max_num=1)
    if len(bboxes) == 0:
        return "Unknown"
    
    # SAFETY CHECK: no face or no landmarks
    if len(bboxes) == 0 or kpss is None or len(kpss) == 0 or kpss[0] is None:
        print("[WARN] No valid face/landmarks detected in current frame")
        return "Unknown"

    # bbox format: assume [x1, y1, x2, y2, score] or similar
    bbox = bboxes[0].astype(int)
    kps = kpss[0]

    # Ensure coordinates are within frame bounds
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))

    # Crop the face image for saving (this is only to save an image file)
    if x2 <= x1 or y2 <= y1:
        return "Unknown"
    face_img = frame[y1:y2, x1:x2]

    # 2) compute embedding for recognition on the live frame
    # Note: recognizer.get_embedding may expect the original frame + landmarks.
    # If it needs full-frame landmarks, pass frame and kps. If it expects face patch,
    # you'd need landmarks relative to the patch. Here we try the robust approach:
    # 2) compute embedding for recognition on the live frame
    embedding = recognizer.get_embedding(frame, kps)
    if embedding is None:
        print("[WARN] Could not compute embedding from full frame + landmarks. Trying cropped face...")
        try:
            embedding = recognizer.get_embedding(face_img, None)
        except Exception as e:
            print(f"[ERROR] Fallback embedding failed: {e}")
            embedding = None

    if embedding is None:
        print("[ERROR] No valid embedding could be computed for this face")
        return "Unknown"
  
    # 3) search DB for a match (find_match already wraps threshold logic)
    name, score = face_db.find_match(embedding, threshold=threshold)
    if name is not None and name != "Unknown":
        # recognized
        with open("recognized_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{name},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return name

    # 4) Unknown -> ask user (interactive)
    new_name = input("Unknown person detected! Enter name to add to database (or leave blank to ignore): ").strip()
    if not new_name:
        return "Unknown"  # user chose not to add

    # 5) Save the cropped face to assets/faces/<new_name> as a persistent sample
    person_dir = os.path.join(faces_dir, new_name)
    os.makedirs(person_dir, exist_ok=True)
    # Use .jpg extension and unique filename
    existing_imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    img_idx = len(existing_imgs) + 1
    save_path = os.path.join(person_dir, f"img_{img_idx}.jpg")
    # Write face image (the cropped face)
    cv2.imwrite(save_path, face_img)

    # 6) Re-load saved image and compute embedding *from that file* (ensures consistency)
    saved_img = cv2.imread(save_path)
    if saved_img is None:
        print(f"[ERROR] Could not read saved image {save_path}")
        return "Unknown"

    # Detect face and landmarks inside the saved image
    saved_bboxes, saved_kpss = detector.detect(saved_img, max_num=1)
    if len(saved_bboxes) == 0:
        print(f"[WARN] No face detected in saved image {save_path}")
        return "Unknown"

    # Always use landmarks from detection
    saved_embedding = recognizer.get_embedding(saved_img, saved_kpss[0])
    if saved_embedding is None:
        print(f"[WARN] Could not compute embedding from saved image {save_path}")
        return "Unknown"


    # 7) Add to in-memory DB and persist the DB to disk
    face_db.add_face(saved_embedding, new_name)
    try:
        face_db.save()
    except Exception as e:
        print(f"[WARN] could not save face database to disk: {e}")

    # Log new face
    with open("new_faces_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{new_name},{save_path},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[INFO] Added '{new_name}' to face database")
    return new_name
