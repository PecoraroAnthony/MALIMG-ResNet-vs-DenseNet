# ------------------------------
# train.py (main script)
# ------------------------------
import os
# Limit TensorFlow logging to only show errors
# 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = only ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
from model_builder import get_model
from data_loader import get_data_generators
from evaluation import evaluate_model
from callbacks import get_callbacks
from tensorflow.keras import mixed_precision

# Optimizes GPU memory usage (default behavior is to allocate all memory)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"---Memory growth enabled for {len(gpus)} GPU(s)---")
    except RuntimeError as e:
        print(f"Could not set memory growth: {e}")

# Define models (comment out the ones you don't want to train)
model_names = [
  # 'resnet50',
  # 'resnet101',
  # 'resnet152',
  # 'densenet121',
  # 'densenet169',
  'densenet201'
]

# Loop over each model
for model_name in model_names:
  print(f"\n\nStarting training for {model_name}...\n")

  # Build the model
  model, preprocess = get_model(model_name, pretrained=False)

  print(f"{model_name} loaded successfully.\n")

  # Load the data generators 
  train_gen, val_gen, test_gen = get_data_generators(
      '../malimg_dataset/train',
      '../malimg_dataset/val',
      '../malimg_dataset/test',
      preprocess_input=preprocess
  )

  # Compile the model
  model.compile(
      optimizer='adam', # Adam optimizer
      loss='categorical_crossentropy', 
      metrics=['accuracy'] 
  )

  # Train the model
  history = model.fit(
      train_gen,
      validation_data=val_gen,
      epochs=40,
      callbacks=get_callbacks(model_name)
  )

  # Evaluate the model and save confusion matrix
  evaluate_model(
      model,
      test_gen,
      class_names=list(train_gen.class_indices.keys()),
      model_name=model_name
  )

  print(f"\n\nFinished training and evaluation for {model_name}!\n\n")
