import time

from utils.error_functions import calculate_prediction_results_nb
from utils.file_operations import get_train_validation_test_csv_files
from utils.normalise_data import DataNormaliser

sorted_grouped_job_files_dir = "/home/anna/Documents/time-series-optimisation/data/sortedGroupedJobFiles/"


def get_naive_prediction_results(sequence_length, t, test_datasets, target):
    prediction_values = list()
    actual_values = list()
    for test_dataset in test_datasets:
        # in a naive model - the prediction = the last actual value of the sequence
        start_train_index = sequence_length - 1
        prediction_value = test_dataset[target][start_train_index + t:]
        prediction_values.append(prediction_value)
        # actual results needs to have the same size as the prediction
        actual_value = test_dataset[target][:-t]
        actual_values.append(actual_value)
    return prediction_values, actual_values


def main(t, sequence_length, target, features):
    start_time = time.time()
    data_normaliser = DataNormaliser()

    training_files_csv, validation_files_csv, test_files_csv = get_train_validation_test_csv_files(
        sorted_grouped_job_files_dir, data_normaliser)

    for current_t in range(t):
        current_t = current_t + 1
        print(current_t)
        pred_train, act_train = get_naive_prediction_results(sequence_length, current_t, training_files_csv,
                                                             target)
        pred_validation, act_validation = get_naive_prediction_results(sequence_length, current_t,
                                                                       validation_files_csv,
                                                                       target)
        pred_test, act_test = get_naive_prediction_results(sequence_length, current_t, test_files_csv, target)

        calculate_prediction_results_nb(current_t, pred_train, act_train, start_time, start_time,
                                        "../results/nb_results_mem/train" + str(current_t) + ".txt")
        calculate_prediction_results_nb(current_t, pred_test, act_test, start_time, start_time,
                                        "../results/nb_results_mem/test" + str(current_t) + ".txt")
        calculate_prediction_results_nb(current_t, pred_validation, act_validation, start_time, start_time,
                                        "../results/nb_results_mem/validation" + str(current_t) + ".txt")


if __name__ == "__main__":
    main(6, 1, 'canonical_mem_usage', 'canonical_mem_usage')
