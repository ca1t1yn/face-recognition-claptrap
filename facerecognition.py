import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv

base_options = python.BaseOptions(model_asset_path="detector.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

base_options_landmarker = python.BaseOptions(model_asset_path="face_landmarker.task")
options_landmarker = vision.FaceLandmarkerOptions(base_options=base_options_landmarker)
detector_landmarker = vision.FaceLandmarker.create_from_options(options_landmarker)

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Error opening camera.")
    exit()

frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))
print("Recording. Press q to exit.")

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
            print(detection.bounding_box)
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv.rectangle(frame, (x, y), (w + x, y + h), (255, 0, 0), 2)

        lm_results = detector_landmarker.detect(mp_image)
        if lm_results.face_landmarks:
            for landmark in lm_results.face_landmarks[0]:
                cx = int(landmark.x*frame_width)
                cy = int(landmark.y*frame_height)
                cv.circle(frame, (cx, cy), 1, (255, 255, 0), -1)

    out.write(frame)
    cv.imshow("Live Recording", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
out.release()
cv.destroyAllWindows()
print("Video saved as 'output.mp4'")
