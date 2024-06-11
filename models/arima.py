import concurrent
import time
from multiprocessing import Process

from utils.deep_learning_utils import set_seed
from utils.file_operations import get_train_validation_test_csv_files
from utils.normalise_data import DataNormaliser
from utils.train_test_predict import predict_arima


def predict_and_save(files_csv, target, t, sequence_length, start_time, training_time, result_dir, order):
    predict_arima(files_csv, target, t, sequence_length, start_time, training_time, result_dir, order)
    print(f"predicted {result_dir}")


def main(t=1, sequence_length=12, features=['mean_CPU_usage'], target=["mean_CPU_usage"], order=(1, 0, 0)):
    sorted_grouped_job_files_dir = "/home/anna/Documents/time-series-optimisation/data/sortedGroupedJobFiles/"
    set_seed()
    start_time = time.time()
    data_normaliser = DataNormaliser()
    training_files_csv, validation_files_csv, test_files_csv = get_train_validation_test_csv_files(
        sorted_grouped_job_files_dir, data_normaliser)
    training_time = round((time.time() - start_time), 2)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use executor.submit to submit tasks individually
        futures = []
        for data_files_csv, result_dir_suffix in zip([training_files_csv, test_files_csv, validation_files_csv],
                                                     ['train', 'test', 'validation']):
            result_dir = f"../results/arima_results_{order}/" + result_dir_suffix
            future = executor.submit(predict_and_save, data_files_csv, target, t, sequence_length, start_time,
                                     training_time, result_dir, order)
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def execute_main(order):
    main(t=6, sequence_length=6, features=['mean_CPU_usage'], target=['mean_CPU_usage'], order=order)


if __name__ == "__main__":
    orders = [(0, 0, 1), (1, 0, 1), (1, 0, 0)]
    processes = []

    for order in orders:
        process = Process(target=execute_main, args=(order,))
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
