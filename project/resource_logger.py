# ------------------------------
# resource_logger.py
# ------------------------------
from tensorflow.keras.callbacks import Callback
import GPUtil, psutil

class ResourceLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"\n[Epoch {epoch+1}] GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB used | Load: {gpu.load*100:.1f}%")
        mem = psutil.virtual_memory()
        print(f"\n[Epoch {epoch+1}] CPU RAM: {mem.used / 1e6:.2f}MB used / {mem.total / 1e6:.2f}MB total")
