# ------------------------------
# isgpu.py (checks if tensorflow recognizes the GPU)
# ------------------------------
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))