# ------------------------------
# eda.py (plots class distribution and sample images)
# ------------------------------
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

def plots(ims, labels, class_indices, cols=10, interp=False):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))

    # Rescale [0,1] back to [0,255]
    ims = (ims * 255).astype(np.uint8)

    idx_to_class = {v: k for k, v in class_indices.items()}

    num_images = min(50, len(ims))
    sampled_indices = random.sample(range(len(ims)), num_images)
    
    # Calculate needed rows
    rows = (num_images + cols - 1) // cols

    # Set high-res figure size (scale up for better visualization)
    fig_w = cols 
    fig_h = rows 

    plt.figure(figsize=(fig_w, fig_h))

    for i, idx in enumerate(sampled_indices):
        sp = plt.subplot(rows, cols, i + 1)
        sp.axis('off')
        label_idx = np.argmax(labels[idx])
        sp.set_title(idx_to_class[label_idx], fontsize=8)
        plt.imshow(ims[idx], interpolation=None if interp else 'none')

    plt.tight_layout()
    os.makedirs("eda", exist_ok=True)
    plt.savefig("eda/sample_images.png", bbox_inches='tight')
    plt.close()

def plot_class_distribution(labels, class_indices):
    class_names = list(class_indices.keys())
    label_indices = np.argmax(labels, axis=1)
    counts = np.bincount(label_indices)

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, counts)
    plt.title("Class Distribution")
    plt.xlabel("Malware Family")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot to a file
    os.makedirs("eda", exist_ok=True)
    plt.savefig("eda/class_distribution.png")
    plt.close()

# main
if __name__ == "__main__":
    # Paths
    dataset_dir = "../malimg_dataset/train"

    # Load data
    datagen = ImageDataGenerator(rescale=1./255)
    batches = datagen.flow_from_directory(
        dataset_dir,
        target_size=(64, 64),
        batch_size=10000,
        class_mode='categorical',
        shuffle=False
    )

    imgs, labels = next(batches)

    # Generate plots
    plots(imgs, labels, batches.class_indices)
    plot_class_distribution(labels, batches.class_indices)

    print("EDA plots saved in 'eda/' folder.")
