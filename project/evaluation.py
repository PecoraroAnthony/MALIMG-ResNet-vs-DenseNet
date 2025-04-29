from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_model(model, test_gen, class_names, model_name):
    y_pred = model.predict(test_gen)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Extract true labels manually from tf.data.Dataset
    y_true = []
    for batch in test_gen:
        images, labels = batch
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)

    print(classification_report(y_true, y_pred_labels, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    os.makedirs("confusion_matrices", exist_ok=True)
    plt.savefig(f"confusion_matrices/{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"Confusion matrix saved to 'confusion_matrices/{model_name}_confusion_matrix.png'")
