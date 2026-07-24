from ultralytics import YOLO

paths = [
    r"C:\Users\cherr\Desktop\DL.project\runs\detect\runs\Defect_Detection-15\weights\best.pt",
    r"C:\Users\cherr\Desktop\DL.project\runs\detect\runs\Defect_Detection-18\weights\best.pt",
    r"C:\Users\cherr\Desktop\DL.project\runs\detect\runs\Defect_Detection-21\weights\best.pt",
    r"C:\Users\cherr\Desktop\DL.project\runs\detect\runs\Defect_Detection-23\weights\best.pt",
    r"C:\Users\cherr\Desktop\DL.project\runs\detect\runs\Defect_Detection-28\weights\best.pt",
]

for p in paths:
    print("\n", p)
    model = YOLO(p)
    print(model.names)