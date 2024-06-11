import warnings
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import torch
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from utils.error_functions import loss_fn, calculate_prediction_results
from statsmodels.tsa.seasonal import seasonal_decompose


def test_model(data_loader, model, device, t):
    model.eval()
    loss_cpu = 0
    with torch.no_grad():  # do not calculate the gradient
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            cpu = model(X)
            desired_shape = (len(cpu), t)
            actual_cpu = y[..., 0]
            actual_cpu = actual_cpu.view(desired_shape)

            loss_cpu += loss_fn(cpu, actual_cpu)
    loss = (loss_cpu / len(data_loader)).item()
    return loss


def train_model(data_loader, model, optimizer, device, t):
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        cpu = model(X)
        desired_shape = (len(cpu), t)  # should be same as batch size, but in case data%batch size isn't 0, we need this
        actual_cpu = y[..., 0]
        actual_cpu = actual_cpu.view(desired_shape)

        loss = loss_fn(cpu, actual_cpu)
        loss.backward()
        optimizer.step()


def train_hybrid_model(data_loader, model, optimizer, device, t):
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        result = seasonal_decompose(X, model='additive', period=288)  # daily seasonality
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        cpu = model(X)
        desired_shape = (len(cpu), t)  # should be same as batch size, but in case data%batch size isn't 0, we need this
        actual_cpu = y[..., 0]
        actual_cpu = actual_cpu.view(desired_shape)

        loss = loss_fn(cpu, actual_cpu)
        loss.backward()
        optimizer.step()


def train_transformer_model(data_loader, model, optimizer, device, t):
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        cpu = model(X, y)
        desired_shape = (len(cpu), t)  # should be same as batch size, but in case data%batch size isn't 0, we need this
        actual_cpu = y[..., 0]
        actual_cpu = actual_cpu.view(desired_shape)

        loss = loss_fn(cpu, actual_cpu)
        loss.backward()
        optimizer.step()


def predict(data_loader, model, device):
    cpu = torch.empty(0, device=device)  # Initialize an empty tensor on the desired device
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_prediction_cpu = model(X)
            cpu = torch.cat((cpu, y_prediction_cpu), 0)
    return cpu


def predict_rf(files_csv_x, files_csv_y, t, sequence_length, start_time, training_time, regressor, result_path):
    files_csv_x_reshaped = [np.reshape(arr, (-1, sequence_length)).tolist() for arr in files_csv_x]
    y_predictions = list()
    for files_csv_x in files_csv_x_reshaped:
        y_prediction = regressor.predict(files_csv_x)
        y_predictions.append(y_prediction)

    calculate_prediction_results(t, y_predictions, files_csv_y, start_time,
                                 training_time, result_path)


def predict_arima(files_csv, target, t, sequence_length, start_time, training_time, result_path, order):
    for csv in files_csv:
        indices = pd.DatetimeIndex(csv["start_time"])
        indices = indices.tz_localize(timezone.utc).tz_convert('US/Eastern')
        first_timestamp = indices[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
        increment = timedelta(minutes=5)
        indices = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
                   [first_timestamp + i * increment for i in range(len(indices))]]

        history_indices = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
                           reversed([first_timestamp - i * increment for i in range(1, sequence_length + 1)])]

        csv['start_time'] = indices
        csv.set_index('start_time', inplace=True)
        data = csv.loc[:, [target[0]]]
        predictions = []
        observations = []
        history = pd.DataFrame(index=pd.to_datetime(history_indices))
        history[target] = 0
        for x in range(int(len(data) - t + 1)):
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                model = ARIMA(history, order=order)
                model_fit = model.fit()
                output = model_fit.forecast(steps=t)
            except np.linalg.LinAlgError as e:
                print("Error occurred:", e)
                predictions.append([0, 0, 0, 0, 0, 0])
            else:
                if isinstance(output, pd.Series):
                    output = output.values
                predictions.append(output)
            finally:
                observations.append(np.squeeze(data.iloc[x:x + sequence_length][target].values))
                history = pd.concat([history, pd.DataFrame([data.iloc[x]], columns=history.columns)])
                history = history.iloc[1:]  # remove first row
        calculate_prediction_results(t, [np.array(predictions)], [np.array(observations)], start_time,
                                    training_time, result_path)