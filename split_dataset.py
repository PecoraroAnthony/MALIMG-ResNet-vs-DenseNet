#--------------------------------------------------------------------------------
# This script splits the Malimg dataset into training, validation, and test sets.
#--------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
import os, shutil
from pathlib import Path

source_dir = Path("malimg_raw")  # archive version of datset
target_base = Path("malimg_dataset")  # where train/val/test folders will go

# Loop through each class directory
for class_dir in source_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.png"))  # or .jpg
        train, temp = train_test_split(images, test_size=0.3, random_state=42) # 70% train, 30% temp
        val, test = train_test_split(temp, test_size=0.5, random_state=42) # 15% val, 15% test

        # Create directories for train, val, test
        for subset_name, subset in zip(["train", "val", "test"], [train, val, test]):
            subset_class_dir = target_base / subset_name / class_dir.name
            subset_class_dir.mkdir(parents=True, exist_ok=True)
            for img in subset:
                shutil.copy(img, subset_class_dir / img.name)