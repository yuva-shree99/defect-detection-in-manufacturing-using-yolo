import os
import shutil

# Change this path if needed
base_path = r"C:\Users\cherr\Desktop\DL.project\DL.dataset"

for split in ["train", "validation"]:
    images_path = os.path.join(base_path, split, "images")

    # Loop through class folders
    for folder in os.listdir(images_path):
        folder_path = os.path.join(images_path, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                src = os.path.join(folder_path, file)
                dst = os.path.join(images_path, file)

                shutil.move(src, dst)

            # Remove empty folder
            os.rmdir(folder_path)

print("Images moved successfully!")