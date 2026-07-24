from ultralytics import YOLO

# Load YOLOv8 Nano pretrained model

model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=r"C:\Users\cherr\Desktop\DL.project\data.yaml",   # Path to data.yaml
    epochs=10,                  # Number of training epochs
    imgsz=416,                  # Image size
    batch=16,                   # Batch size
    project="runs",             # Output folde\
    name="Defect_Detection",    # Experiment name
    save=True                   # Save best \
)

print("Training Completed Successfully!")