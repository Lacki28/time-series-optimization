import math
import time

import numpy as np
import sklearn.metrics as sm
import torch

from utils.file_operations import append_to_file


def mse(prediction, real_value):
    MSE = torch.square(torch.subtract(real_value, prediction)).mean()
    return MSE


def naive_ratio(t, prediction, real_value):
    # Compute the absolute difference between corresponding elements of a and b
    prediction_nr = prediction[t:]
    real_value_nr = real_value[:-t]
    abs_diff_et1 = torch.abs(prediction_nr - real_value_nr)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = torch.sum(abs_diff_et1)
    et1 = (1 / len(prediction_nr)) * sum_abs_diff_et1
    abs_diff = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = torch.sum(abs_diff)
    et = (1 / len(prediction)) * sum_abs_diff
    return et / et1


def loss_fn(output, target):
    loss = mse(output, target)
    return loss


def calculate_prediction_results(t, prediction, actual_values, start_time, training_time, path):
    for job_index in range(len(prediction)):
        for i in range(t):
            current_act_cpu_validation = actual_values[job_index][:, i]
            current_pred_cpu_validation = prediction[job_index][:, i]
            if isinstance(current_act_cpu_validation, np.ndarray):
                calc_Latex_results(t, current_act_cpu_validation, current_pred_cpu_validation,
                                   path + str(i + 1) + ".txt", start_time, training_time)
            else:
                calc_Latex_results(t, current_act_cpu_validation.cpu(), current_pred_cpu_validation.cpu(),
                                   path + str(i + 1) + ".txt", start_time, training_time)


def print_prediction_results(t, prediction, actual_values, start_time, training_time, path):
    for job_index in range(len(prediction)):
        for i in range(t):
            current_act_cpu_validation = actual_values[job_index][:, i]
            current_pred_cpu_validation = prediction[job_index][:, i]
            calc_Latex_results(t, current_act_cpu_validation.cpu(), current_pred_cpu_validation.cpu(),
                               path + str(i + 1) + ".txt", start_time, training_time)


def calculate_prediction_results_nb(t, prediction, actual_values, start_time, training_time, path):
    for job_index in range(len(actual_values)):
        current_act_cpu_train = actual_values[job_index].values
        current_act_cpu_train = torch.tensor(current_act_cpu_train)
        current_pred_cpu_train = prediction[job_index].values
        current_pred_cpu_train = torch.tensor(current_pred_cpu_train)
        calc_Latex_results(t, current_act_cpu_train, current_pred_cpu_train, path, start_time, training_time)


def calc_Latex_results(t, y_test, y_test_pred, file_path, start_time, training_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    rmse = math.sqrt(mse)
    if isinstance(y_test, np.ndarray):
        y_test = torch.tensor(y_test)
        y_test_pred = torch.tensor(y_test_pred)

    #    append_to_file(file_path, "mae & mse & rmse & nr & training & total")
    append_to_file(file_path,
                   str(mae) + " & " + str(mse) + " & " + str(rmse) + " & " + str(training_time) + " & " + str(
                       round((time.time() - start_time), 2)))
