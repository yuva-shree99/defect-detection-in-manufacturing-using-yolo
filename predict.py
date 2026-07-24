from ultralytics import YOLO

model = YOLO(
    r"C:\Users\cherr\Desktop\DL.project\runs\detect\runs\Defect_Detection-28\weights\best.pt"
)

results = model.predict(
    source=r"C:\Users\cherr\Desktop\DL.project\DL.dataset\validation\images\scratches_300.jpg",
    conf=0.01,
    imgsz=640,
    verbose=True,
    save=True
)

print("Boxes:", results[0].boxes)
print("Classes:", results[0].boxes.cls)
print("Confidence:", results[0].boxes.conf)