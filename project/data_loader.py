# ------------------------------
# data_loader.py (Loads and preprocesses the dataset)
# ------------------------------
import tensorflow as tf

# Preprocesses the images such that they are standardized to be 64x64 pixels and batched into groups of 128 using tf.data pipeline

def get_data_generators(train_dir, val_dir, test_dir, preprocess_input, target_size=(64, 64), batch_size=128):
  AUTOTUNE = tf.data.AUTOTUNE

  def preprocess_image(image, label):
      image = tf.image.resize(image, target_size)
      image = preprocess_input(image)
      return image, label

  def prepare_dataset(dataset, training=True):
      if training:
          dataset = dataset.shuffle(1000)
      dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(buffer_size=AUTOTUNE)
      return dataset

  train_ds = tf.keras.utils.image_dataset_from_directory(
      train_dir,
      labels='inferred',
      label_mode='categorical',
      image_size=target_size,
      shuffle=True,
      batch_size=None
  )

  val_ds = tf.keras.utils.image_dataset_from_directory(
      val_dir,
      labels='inferred',
      label_mode='categorical',
      image_size=target_size,
      shuffle=False,
      batch_size=None
  )

  test_ds = tf.keras.utils.image_dataset_from_directory(
      test_dir,
      labels='inferred',
      label_mode='categorical',
      image_size=target_size,
      shuffle=False,
      batch_size=None
  )

  # Capture class names here
  class_names = train_ds.class_names

  # Preprocess and optimize datasets
  train_gen = prepare_dataset(train_ds, training=True)
  val_gen = prepare_dataset(val_ds, training=False)
  test_gen = prepare_dataset(test_ds, training=False)

  return train_gen, val_gen, test_gen, class_names