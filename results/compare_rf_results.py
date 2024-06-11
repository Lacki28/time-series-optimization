import os
import statistics

import numpy as np
from matplotlib import pyplot as plt


class Config:
    dir = None
    title = None
    figure_name = None

    def __init__(self, mode, start, end):
        self.mode = mode
        if self.mode == 'rf':
            self.dir = "./hyperparameter_results_new/rf/rf_results_cpu_tests"
            self.title = "RF"
            self.figure_name = "RF"
            self.start = start
            self.end = end
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
    for i in (8, 16, 24, 32):#range(config.start, config.end+1):
        directory= config.dir+"_"+str(i)+"_100/"
        for timestamp in (1, 6):
            print(directory)
            with open(directory+"statistics/" + data_set + str(timestamp) + "_statistics" + ".txt", 'r') as file:
                line = file.readline()
                line = file.readline()
                # for line in file:
                test_error_values = line.split(' & ')
                if len(test_error_values)!=1:
                    mae.append(float(test_error_values[0]))
                    mse.append(float(test_error_values[1]))
                    rmse.append(float(test_error_values[2]))
    print(mse)


def main():
    config = Config("rf",6, 10)
    get_avg_loss(config, "test")
    get_avg_loss(config,"validation")
    # get_avg_loss(config, "test")


if __name__ == "__main__":
    main()
