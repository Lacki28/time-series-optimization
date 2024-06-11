import os
import statistics

import numpy as np
from matplotlib import pyplot as plt

import shlex
class Config:
    dir = None
    title = None
    figure_name = None

    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'arima':
            self.dir = "./arima_tests/arima_results"
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
    plt.title(config.title +" "+ loss_name, fontsize=22)
    plt.savefig(config.dir +"statistics/"+ config.figure_name +"_"+ data_set + '_' + loss_name + '_loss' + '.png')
    plt.close()


def calc_avg(lst):
    result = sum(lst) / len(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)


def calc_std(lst):
    result = statistics.stdev(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)


def get_avg_loss(config, data_set):
    mae = []
    mse = []
    rmse = []
    orders = ["001", "010", "011", "100", "101", "110", "111" ]
    for i in orders:#range(config.start, config.end+1):
        directory = f"{config.dir}_{i}/"
        # for timestamp in range(6):
        absolute_path = os.path.abspath(directory+"statistics/" + data_set + str(5 + 1) + "_statistics"+ ".txt")
        try:
            with open(absolute_path, 'r') as file:
                for line in file:
                    test_error_values = line.split(' & ')
                    if len(test_error_values)!=1:
                        mae.append(float(test_error_values[0]))
                        mse.append(float(test_error_values[1]))
                        rmse.append(float(test_error_values[2]))
        except FileNotFoundError:
            print(f"The specified file '{absolute_path}' could not be found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    print(mse)


def main():
    config = Config("arima")
    get_avg_loss(config,"validation")
    get_avg_loss(config, "train")
    # get_avg_loss(config, "test")


if __name__ == "__main__":
    main()
