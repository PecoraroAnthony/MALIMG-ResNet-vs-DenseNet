# ------------------------------
# graph_classification.py (generates bar graphs from classification reports)
# ------------------------------
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Directory containing classification reports
REPORT_DIR = "reports"

# Define model order to match the bar graph layout (excluding LeNet-5)
model_order = [
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet169",
    "densenet201"
]

# Display labels
model_labels = {
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "densenet121": "DenseNet-121",
    "densenet169": "DenseNet-169",
    "densenet201": "DenseNet-201"
}

# Function to extract metrics from classification report
def extract_metrics(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    accuracy_match = re.search(r'accuracy\s+([\d.]+)', text)
    precision_match = re.search(r'macro avg\s+([\d.]+)', text)
    recall_match = re.search(r'macro avg\s+[\d.]+\s+([\d.]+)', text)
    f1_match = re.search(r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', text)

    if accuracy_match and precision_match and recall_match and f1_match:
        return {
            'accuracy': float(accuracy_match.group(1)),
            'precision': float(precision_match.group(1)),
            'recall': float(recall_match.group(1)),
            'f1': float(f1_match.group(1)),
        }
    return None

# Collect metrics for each model
metrics = {}
for filename in os.listdir(REPORT_DIR):
    model_name = filename.replace("_classification_report.txt", "").lower()
    filepath = os.path.join(REPORT_DIR, filename)
    values = extract_metrics(filepath)
    if values:
        metrics[model_name] = values

# Organize data for plotting
bar_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
bar_width = 0.12
x = np.arange(len(bar_labels))

plt.figure(figsize=(12, 6))

for idx, model in enumerate(model_order):
    if model in metrics:
        values = [
            metrics[model]['accuracy'],
            metrics[model]['precision'],
            metrics[model]['recall'],
            metrics[model]['f1'],
        ]
        plt.bar(x + idx * bar_width, values, width=bar_width, label=model_labels[model])

plt.ylabel('%')
plt.xticks(x + bar_width * (len(model_order) - 1) / 2, bar_labels)
plt.ylim(0.0, 1.05)
plt.legend()
plt.title("Model Comparison: Accuracy, Precision, Recall, F1")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()