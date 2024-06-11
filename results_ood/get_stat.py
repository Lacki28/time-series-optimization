import os
import statistics

import numpy as np
from matplotlib import pyplot as plt


class Config:
    dir = None
    title = None
    figure_name = None

    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'nb':
            self.dir = "./nb_results_mem/"
            self.title = "Naive Benchmark"
            self.figure_name = "NB"
        elif self.mode == 'lstm':
            self.dir = "lstm_results_mem_final/"
            self.title = "LSTM"
            self.figure_name = "LSTM"
        elif self.mode == 'transformer':
            self.dir = "transformer_encoder_results_mem_final/"
            self.title = "Transformer"
            self.figure_name = "Transformer"
        elif self.mode == 'rf':
            self.dir = "./rf_results_mem_16_200/"
            self.title = "RF"
            self.figure_name = "RF"
        elif self.mode == 'arima':
            self.dir = "./arima_results_(0, 0, 1)/"
            self.title = "ARIMA"
            self.figure_name = "ARIMA"
        else:
            raise ValueError("Invalid mode")


def calculate_outliers(data):
    for inner_list in data:
        data_arr = np.array(inner_list, dtype=np.float64)

        Q1 = np.percentile(data_arr, 25)
        Q3 = np.percentile(data_arr, 75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(len(data_arr))
        print(len([x for x in data_arr if lower_bound <= x <= upper_bound]))


def create_boxplot(loss, loss_name, config, data_set):
    calculate_outliers(loss)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    plt.boxplot(loss, showfliers=False)
    plt.xlabel('Timestamps ahead', fontsize=20)
    plt.ylabel('Metric ' + loss_name, fontsize=20)
    plt.title(config.title + " " + loss_name, fontsize=22)
    plt.savefig(config.dir + "statistics/" + config.figure_name + "_" + data_set + '_' + loss_name + '_loss' + '.png')
    plt.close()


def calc_avg(lst):
    result = sum(lst) / len(lst)
    formatted_result = "{:.4f}".format(result)
    return str(formatted_result)


def calc_std(lst):
    result = statistics.stdev(lst)
    formatted_result = "{:.4f}".format(result)
    return str(formatted_result)


def calc_median(lst):
    lst.sort()
    median_value = statistics.median(lst)
    rounded_median = round(median_value, 6)
    return str(rounded_median)

def get_avg_loss(directory, data_set):
    list_of_mae = []
    list_of_mse = []
    list_of_rmse = []
    list_of_nr = []
    list_of_total_time = []
    for timestamp in range(6):
        mae = []
        mse = []
        rmse = []
        nr = []
        total_time = []
        with open(directory + data_set + str(timestamp + 1) + ".txt", 'r') as file:
            for line in file:
                test_error_values = line.split(' & ')
                mae.append(float(test_error_values[0]))
                mse.append(float(test_error_values[1]))
                rmse.append(float(test_error_values[2]))
                nr.append(float(test_error_values[3]))
                total_time.append(float(test_error_values[4]))
        directory_path = os.path.join(directory, "statistics_new")
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(directory + "statistics_new/" + data_set + "_statistics" + ".txt", 'a+') as file:
            # file.write(calc_avg(mse) + " & " + calc_std(mse) + " & ")
            file.write(" \n")
            file.write(calc_median(mse) + " & ")
            # file.write("std" + " \n")
            # file.write(calc_std(mae) + " & " + calc_std(mse) + " & " + calc_std(rmse) + " \n")
            # file.write(calc_avg(total_time) + " \n")
        list_of_mae.append(mae)
        list_of_mse.append(mse)
        list_of_rmse.append(rmse)
        list_of_nr.append(nr)
        list_of_total_time.append(total_time)
    # create_boxplot(list_of_mae, "MAE", config, data_set)
    # create_boxplot(list_of_mse, "MSE", config, data_set)
    # create_boxplot(list_of_rmse, "RMSE", config, data_set)


def main():
    # get_avg_loss("lstm_results_cpu_final/", "train")
    # get_avg_loss("lstm_results_cpu_final/", "validation")
    # get_avg_loss("lstm_results_cpu_final/", "test")
    cpu_list = ["./nb_results_cpu/", "./arima_results_cpu_(1, 0, 0)/", "./rf_results_cpu_FINAL_16_200/",
                "lstm_results_cpu_FINAL/",
                "transformer_results_cpu_FINAL/"]

    mem_list = ["./nb_results_mem/", "./arima_results_mem_(1, 0, 0)/", "./rf_results_mem_FINAL_16_200/",
                "lstm_results_mem_FINAL/", "transformer_results_mem_FINAL/"]
    for directory in mem_list:
        get_avg_loss(directory, "test")
    for directory in cpu_list:
        get_avg_loss(directory, "test")

if __name__ == "__main__":
    main()
