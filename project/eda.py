# ------------------------------
# eda.py
# ------------------------------
import matplotlib.pyplot as plt
import numpy as np

def plots(ims, labels, class_indices, figsize=(20, 30), rows=10, cols=10, interp=False):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    class_names = list(class_indices.keys())
    for i in range(min(rows * cols, len(ims))):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('off')
        label_idx = np.argmax(labels[i])
        sp.set_title(class_names[label_idx], fontsize=8)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.tight_layout()
    plt.show()