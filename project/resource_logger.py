# ------------------------------
# resource_logger.py (generates CSV logs for resource usage during training)
# ------------------------------
from tensorflow.keras.callbacks import Callback
import psutil
import os
import subprocess
import csv
import time

class ResourceLogger(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        os.makedirs("logs", exist_ok=True)
        self.csv_path = f"logs/{self.model_name}_resource_usage.csv"
        self.total_time = 0
        self.epoch_times = []

        # Write CSV header
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "GPU_Memory_MB", "CPU_RAM_Used_MB", "CPU_RAM_Total_MB", "Epoch_Training_Time_Seconds"])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.total_time += epoch_time
        self.epoch_times.append(epoch_time)

        pid = os.getpid() # get user process id
        gpu_memory = 0

        # GPU Usage
        try:
            # Capture the output of nvidia-smi by querying using pid
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            gpu_usage_lines = result.stdout.strip().split('\n')
            for line in gpu_usage_lines:
                process_pid, mem_used = line.split(',')
                if int(process_pid.strip()) == pid:
                    gpu_memory = int(mem_used.strip())
                    break
        except Exception as e:
            print(f"Error querying nvidia-smi: {e}")

        # CPU RAM Usage
        mem = psutil.virtual_memory()
        cpu_ram_used = mem.used / 1e6  # MB
        cpu_ram_total = mem.total / 1e6  # MB

        # Save to CSV
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, gpu_memory, cpu_ram_used, cpu_ram_total, epoch_time])

        # Clean output formatting after Keras progress bar
        print("\n")
        print(f"[Epoch {epoch+1}] Resource Usage:")
        print(f"  GPU: {gpu_memory}MB")
        print(f"  CPU RAM: {cpu_ram_used:.2f}MB / {cpu_ram_total:.2f}MB")
        print(f"  Epoch Time: {epoch_time:.2f} sec\n")

    def on_train_end(self, logs=None):
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        print(f"\nTotal training time for {self.model_name}: {self.total_time:.2f} seconds")
        print(f"Average training time per epoch for {self.model_name}: {avg_epoch_time:.2f} seconds\n")