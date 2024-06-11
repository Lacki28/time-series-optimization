import os

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from utils.file_operations import read_files
from utils.normalise_data import DataNormaliser

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

# Example usage:

def plot_files(csv_files, file_names, file_index, directory_path):
    for df in csv_files:
        df = df.apply(lambda x: savgol_filter(x, 51, 4))
        df = data_normalizer.normalize_data_minMax(["mean_CPU_usage"], df)
        data = df["mean_CPU_usage"]

        # Plot the data
        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        plt.plot(data, color='b', label="mean_CPU_usage")
        plt.xlabel('Mean CPU usage')  # Add appropriate labels to the axes
        plt.ylabel('Number of timestamps')
        plt.title('Mean CPU usage')
        plt.legend()  # Show legend with column name
        plt.grid(True)  # Add gridlines if desired
        plt.savefig(directory_path+"/"+str(file_names[file_index][:-4]) + '.png')
        file_index += 1


sorted_grouped_job_files_dir = "/home/anna/Documents/time-series-optimisation/data/sortedGroupedJobFiles/"

file_list = os.listdir(sorted_grouped_job_files_dir)
file_list.sort()
sorted_file_list = file_list
num_files = len(sorted_file_list)

train_ratio = 0.6
test_ratio = 0.2

train_index = int(train_ratio * num_files)
test_index = train_index + int(test_ratio * num_files)

train_files = sorted_file_list[:train_index]
test_files = sorted_file_list[train_index:test_index]
validation_files = sorted_file_list[test_index:]
data_normalizer = DataNormaliser()
training_files_csv = read_files(train_files, True, sorted_grouped_job_files_dir, data_normalizer)
validation_files_csv = read_files(validation_files, False, sorted_grouped_job_files_dir, data_normalizer)
test_files_csv = read_files(test_files, False, sorted_grouped_job_files_dir, data_normalizer)
directory_path = "../plots"
create_directory(directory_path)
directory_path = "../plots/test"
create_directory(directory_path)
plot_files(test_files_csv, test_files, 0, directory_path)
directory_path = "../plots/validation"
create_directory(directory_path)
plot_files(validation_files_csv, validation_files, 0, directory_path)
directory_path = "../plots/train"
create_directory(directory_path)
plot_files(training_files_csv, train_files, 0, directory_path)