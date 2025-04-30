# ------------------------------
# callbacks.py
# ------------------------------
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from resource_logger import ResourceLogger

def get_callbacks(model_name):
    return [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5),
        ModelCheckpoint(f'models/{model_name}.keras', save_best_only=True),
        ResourceLogger(model_name)
    ]