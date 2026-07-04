import torch
import cv2 as cv
import mediapipe as mp
import numpy as np
import os

from torchvision import transforms
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from facenet_pytorch import InceptionResnetV1

base_options = python.BaseOptions(model_asset_path="detector.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options=options)

base_options_landmarker = python.BaseOptions(model_asset_path="face_landmarker.task")
options_landmarker = vision.FaceLandmarkerOptions(base_options=base_options_landmarker)
detector_landmarker = vision.FaceLandmarker.create_from_options(
    options=options_landmarker
)

resnet = InceptionResnetV1(pretrained="vggface2").eval()
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def add_face(name, embedding):
    if name not in face_database:
        face_database[name] = []
        face_database[name].append(embedding.squeeze().detach().numpy())
        print(f"Enrolled{name}")


def match_face(embedding, threshold=0.6):
    if not face_database:
        return "Unknown", 0.0

    embedding = embedding.squeeze().detach().numpy()
    best_match = "Unknown"
    best_score = -1
    for name, saved_embeddings in face_database.items():
        for saved_embedding in saved_embeddings:
            score = np.dot(embedding, saved_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(saved_embedding)
            )
            if score > best_score:
                best_score = score
            if score > threshold:
                best_match = name
    return best_match, best_score

def delete_face(name):
    if name in face_database:
        del face_database[name]
        print(f"{name} deleted from database")
    else:
        print(f"{name} not found in database")


def save_database(path="face_database.npy"):
    np.save(path, face_database)
    print(f"Databased saved to {path}")


def load_database(path="face_database.npy"):
    global face_database
    if os.path.exists(path):
        face_database = np.load(path, allow_pickle=True).item()
        print("Database Loaded")
    else:
        print("No database found, creating a new one")


face_database = {}
load_database()


cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Error opening Camera.")
    exit(0)

frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))
print("Recording.Press q to exit")
while True:
    ret, frame = cam.read()
    if not ret:
        break
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    if results.detections:
        print(f"Faces detected:{len(results.detections)}")

        for detection in results.detections:
           # print(detection.bounding_box)
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv.rectangle(frame, (x, y), (w + x, y + h), (255, 0, 0), 2)
            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size > 0:
                face_tensor = preprocess(face_crop).unsqueeze(0)
                with torch.no_grad():
                    embedding = resnet(face_tensor)
                name, score = match_face(embedding)
                cv.putText(
                    frame,
                    f"{name}{score:.2f}",
                    (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                print(embedding.shape)

        lm_results = detector_landmarker.detect(mp_image)
        if lm_results.face_landmarks:
            for landmark in lm_results.face_landmarks[0]:
                cx = int(landmark.x * frame_width)
                cy = int(landmark.y * frame_height)
                cv.circle(frame, (cx, cy), 1, (255, 255, 0), -1)

    out.write(frame)
    cv.imshow("Live Recording", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("a"):
        name = input("Enter name of face:")
        add_face(name, embedding)
    if key == ord("q"):
        break
    if key == ord("s"):
        save_database()
    if key==ord("d"):
        name=input("Enter name of face you want to delete:")
        delete_face(name)


cam.release()
out.release()
cv.destroyAllWindows()
print("Video saved as 'output.mp4'")
