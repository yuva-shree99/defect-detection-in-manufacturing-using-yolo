import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\cherr\Desktop\DL.project\runs\detect\runs\Defect_Detection-18\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Run YOLO prediction
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Real-Time Defect Detection", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()