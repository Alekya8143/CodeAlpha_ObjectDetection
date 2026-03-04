from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection + Tracking
    results = model.track(frame, persist=True)

    annotated_frame = results[0].plot()

    # Object count
    count = len(results[0].boxes) if results[0].boxes is not None else 0

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Display info
    cv2.putText(annotated_frame, f"Objects: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(annotated_frame, "CodeAlpha AI Surveillance System",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Object Detection & Tracking", annotated_frame)

    # Press S → save screenshot
    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite("capture.jpg", annotated_frame)
        print("Screenshot saved!")

    # Press Q → quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()