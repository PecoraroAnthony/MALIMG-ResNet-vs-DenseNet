# ------------------------------
# train.py (main script)
# ------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = only ERROR

import tensorflow as tf
from model_builder import get_model
from data_loader import get_data_generators
from evaluation import evaluate_model
from callbacks import get_callbacks
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Memory growth failed: {e}")

# Model names to train
model_names = [
  'resnet50',
  'resnet101',
  'resnet152',
  'densenet121',
  'densenet169',
  'densenet201'
]

# Track model training histories
model_histories = {}

# Loop over each model
for model_name in model_names:
    print(f"\n\nStarting training for {model_name}...\n\n")

    model, preprocess = get_model(model_name, pretrained=False)
    print(f"Model {model_name} loaded successfully.\n")

    # Load data 
    train_gen, val_gen, test_gen, class_names = get_data_generators(
        '../malimg_dataset/train',
        '../malimg_dataset/val',
        '../malimg_dataset/test',
        preprocess_input=preprocess
    )

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model and get history
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=40,
        callbacks=get_callbacks(model_name)
    )

    # Save history
    model_histories[model_name] = history.history

    # Eval model: produce classification report and confusion matrix
    evaluate_model(
        model,
        test_gen,
        class_names=class_names,
        model_name=model_name
    )

    print(f"\n\nFinished training and evaluation for {model_name}!\n\n")

# Create output directory
os.makedirs("training_plots", exist_ok=True)

# Plot validation accuracy comparison
plt.figure(figsize=(12, 8))
for model_name, history in model_histories.items():
    val_acc = history.get('val_accuracy', [])
    epochs = range(1, len(val_acc) + 1)
    plt.plot(epochs, val_acc, label=model_name)
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("training_plots/validation_accuracy_comparison.png", dpi=300)
plt.close()

print("Validation accuracy plot saved at 'training_plots/validation_accuracy_comparison.png'")

# Plot validation loss comparison
plt.figure(figsize=(12, 8))
for model_name, history in model_histories.items():
    val_loss = history.get('val_loss', [])
    epochs = range(1, len(val_loss) + 1)
    plt.plot(epochs, val_loss, label=model_name)
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.ylim(0, 3.0)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("training_plots/validation_loss_comparison.png", dpi=300)
plt.close()

print("Validation loss plot saved at 'training_plots/validation_loss_comparison.png'")


print("---------------------------------------------------")
print("|Training and evaluation completed for all models.|")
print("---------------------------------------------------")