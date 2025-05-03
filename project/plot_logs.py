import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the logs
log_dir = 'logs'

# Filenames and model names
model_files = {
    "resnet50": "resnet50_resource_usage.csv",
    "resnet101": "resnet101_resource_usage.csv",
    "resnet152": "resnet152_resource_usage.csv",
    "densenet121": "densenet121_resource_usage.csv",
    "densenet169": "densenet169_resource_usage.csv",
    "densenet201": "densenet201_resource_usage.csv"
}

# Gather resource and timing data
usage_data = []
for model_name, filename in model_files.items():
    file_path = os.path.join(log_dir, filename)
    try:
        df = pd.read_csv(file_path)
        max_gpu = df['GPU_Memory_MB'].max()
        max_cpu = df['CPU_RAM_Used_MB'].max()
        avg_time = df['Epoch_Training_Time_Seconds'].mean()
        usage_data.append({
            "Model": model_name,
            "Max_GPU_Memory_MB": round(max_gpu, 2),
            "Max_CPU_RAM_Used_MB": round(max_cpu, 2),
            "Avg_Training_Time_Sec": round(avg_time, 2)
        })
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

# Create DataFrame
usage_df = pd.DataFrame(usage_data)

# --- Plot 1: GPU and CPU RAM Usage ---
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(usage_df))

plt.bar(index, usage_df['Max_GPU_Memory_MB'], bar_width, label='Max GPU Memory (MB)')
plt.bar([i + bar_width for i in index], usage_df['Max_CPU_RAM_Used_MB'], bar_width, label='Max CPU RAM Used (MB)')

plt.xlabel('Model')
plt.ylabel('Memory Usage (MB)')
plt.title('Max GPU and CPU RAM Usage per Model')
plt.xticks([i + bar_width / 2 for i in index], usage_df['Model'], rotation=45)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'resource_usage_comparison.png'), dpi=300)
plt.close()

# --- Plot 2: Average Training Time per Epoch ---
plt.figure(figsize=(12, 6))
plt.bar(usage_df['Model'], usage_df['Avg_Training_Time_Sec'], color='steelblue')
plt.xlabel('Model')
plt.ylabel('Avg Training Time per Epoch (s)')
plt.title('Average Epoch Training Time by Model')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'average_training_time_comparison.png'), dpi=300)
plt.close()

print("Plots saved to 'logs/' directory.")
