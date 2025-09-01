import cv2
import os
from face_recognition.main import SCRFD, ArcFace, build_face_database, frame_processor, parse_args

# Initialize models and database
args = parse_args()
detector = SCRFD(args.det_weight, input_size=(640, 640), conf_thres=args.confidence_thresh)
recognizer = ArcFace(args.rec_weight)
face_db = build_face_database(detector, recognizer, args)

# Ensure faces directory exists inside face_recognition/assets/faces
save_dir = os.path.join(os.path.dirname(__file__), "assets", "faces")
os.makedirs(save_dir, exist_ok=True)

def get_person_name(frame, detector, recognizer, face_db, threshold=0.5):
    bboxes, kpss = detector.detect(frame, max_num=1)
    if len(bboxes) == 0:
        return "Unknown"

    bbox = bboxes[0].astype(int)
    landmarks = kpss[0]

    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    face_img = frame[y1:y2, x1:x2]

    embedding = recognizer.get_embedding(face_img, landmarks)
    if embedding is None:
        return "Unknown"

    # --- Check if this face already exists in the database ---
    name, score = face_db.find_match(embedding)  # <--- this is critical
    if name is not None and score < threshold:
        return name  # face recognized

    # --- If not recognized, ask for name ---
    new_name = input("Unknown person detected! Enter name to add to database (or leave blank to ignore): ").strip()
    if not new_name:
        return None

    # Save image and add to database (your existing code)
    person_dir = os.path.join("face_recognition", "assets", "faces", new_name)
    os.makedirs(person_dir, exist_ok=True)
    existing_imgs = [f for f in os.listdir(person_dir) if f.lower().endswith(".jpg")]
    img_idx = len(existing_imgs) + 1
    save_path = os.path.join(person_dir, f"img_{img_idx}.jpg")
    cv2.imwrite(save_path, face_img)
    face_db.add_face(embedding, new_name)
    with open("new_faces_log.txt", "a") as f:
        f.write(f"{new_name},{save_path}\n")

    print(f"[INFO] Added '{new_name}' to face database")
    return new_name
