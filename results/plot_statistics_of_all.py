import os
import statistics

import numpy as np
from matplotlib import pyplot as plt


class Config:
    dir = None
    title = None
    figure_name = None

    def __init__(self):
        self.dir = ["./nb_results_mem/", "./arima_results_mem_(1, 0, 0)/", "./rf_results_mem_32_300/", "./lstm_results_mem/",
                     "transformer_encoder_results_mem/"]
        #self.dir =["./nb_results/", "./arima_results_(1, 0, 0)/", "./rf_results_32_300/", "./lstm_results/",
        # "transformer_encoder_results/"]
        self.title = ["Naïve Benchmark", "ARIMA", "RF", "LSTM", "Transformer"]
        self.figure_name = ["Naïve Benchmark", "ARIMA", "RF", "LSTM", "Transformer"]
        self.x_values = ["NB", "ARIMA", "RF", "LSTM", "Transformer"]


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


def create_boxplot(loss, loss_name, data_set):
    # calculate_outliers(loss)
    print(loss.keys())
    config = Config()
    for i in [0, 3]:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 16))
        fig.subplots_adjust(hspace=0.5)
        for timestamp in range(1, 4):
            # ax[timestamp - 1].set_ylim(top=0.000125)
            values_for_timestamp = [loss[dir + str(timestamp+i)] for dir in config.dir]
            ax[timestamp - 1].boxplot(values_for_timestamp, showfliers=False)
            ax[timestamp - 1].tick_params(axis='y', labelsize=14)
            if(timestamp ==1):
                ax[timestamp - 1].set_title(f'{loss_name} results {timestamp + i} timestamp ahead', fontsize=22)
            else:
                ax[timestamp - 1].set_title(f'{loss_name} results {timestamp + i} timestamps ahead', fontsize=22)
            ax[timestamp - 1].set_xticklabels(config.x_values, rotation=45, ha='right',
                                              fontsize=20)  # Rotate x-axis labels for better visibility
        plt.savefig("./mem/"+loss_name+'_'+data_set + '_boxplot' + str(timestamp + i) + '.png')
        plt.show()


def calc_avg(lst):
    result = sum(lst) / len(lst)
    rounded_result = round(result, 6)
    return str(rounded_result)


def calc_std(lst):
    result = statistics.stdev(lst)
    rounded_result = round(result, 6)
    return str(rounded_result)


def get_avg_loss(data_set):
    list_of_mae = {}
    list_of_mse = {}
    list_of_rmse = {}
    list_of_total_time = {}
    config = Config()
    for dir in config.dir:
        for timestamp in range(1, 7):
            print(timestamp)
            mae = []
            mse = []
            rmse = []
            total_time = []
            with open(dir + data_set + str(timestamp) + ".txt", 'r') as file:
                for line in file:
                    test_error_values = line.split(' & ')
                    mae.append(float(test_error_values[0]))
                    mse.append(float(test_error_values[1]))
                    rmse.append(float(test_error_values[2]))
                    total_time.append(float(test_error_values[4]))
            list_of_mae[dir + str(timestamp)] = mae
            list_of_mse[dir + str(timestamp)] = mse
            list_of_rmse[dir + str(timestamp)] = rmse
            list_of_total_time[dir + str(timestamp)] = total_time
    print(list_of_mse)
    print(len(list_of_mse))
   # create_boxplot(list_of_mae, "MAE", data_set)
    create_boxplot(list_of_mse, "MSE", data_set)
    #create_boxplot(list_of_rmse, "RMSE", data_set)


def main():
    get_avg_loss("train")
    get_avg_loss("validation")
    get_avg_loss("test")


if __name__ == "__main__":
    main()
