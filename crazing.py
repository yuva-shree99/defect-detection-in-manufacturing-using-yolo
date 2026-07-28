from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="crazing_242.jpg",
    conf=0.05,
    save=True
)

print(results[0].boxes)