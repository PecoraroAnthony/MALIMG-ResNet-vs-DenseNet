# ------------------------------
# data_loader.py
# ------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, val_dir, test_dir, preprocess_input, target_size=(64, 64), batch_size=128):
  datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  train_gen = datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical')
  val_gen = datagen.flow_from_directory(val_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical')
  test_gen = datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

  return train_gen, val_gen, test_gen