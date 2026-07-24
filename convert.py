import os
import xml.etree.ElementTree as ET

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

xml_folder = r"C:\Users\cherr\Desktop\DL.project\DL.dataset\train\images"
output_folder = r"C:\Users\cherr\Desktop\DL.project\DL.dataset\train\labels"

os.makedirs(output_folder, exist_ok=True)

for xml_file in os.listdir(xml_folder):

    if xml_file.endswith(".xml"):

        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        yolo_lines = []

        for obj in root.findall("object"):

            name = obj.find("name").text

            if name not in classes:
                continue

            class_id = classes.index(name)

            box = obj.find("bndbox")

            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)

            x_center = ((xmin+xmax)/2)/width
            y_center = ((ymin+ymax)/2)/height
            w = (xmax-xmin)/width
            h = (ymax-ymin)/height

            yolo_lines.append(
                f"{class_id} {x_center} {y_center} {w} {h}"
            )

        txt_name = xml_file.replace(".xml",".txt")

        with open(os.path.join(output_folder,txt_name),"w") as f:
            f.write("\n".join(yolo_lines))

print("Conversion completed")