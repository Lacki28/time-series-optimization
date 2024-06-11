import os
import time
from functools import partial

import torch
import torch.nn as nn
from ray import train
from ray import tune
from ray.train import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.deep_learning_utils import set_seed, get_prediction_results, \
    get_train_validation_data_loaders, get_test_data
from utils.error_functions import calculate_prediction_results
from utils.file_operations import append_to_file, get_train_validation_test_csv_files
from utils.normalise_data import DataNormaliser
from utils.train_test_predict import train_model, test_model


class RegressionLSTM(nn.Module):
    def __init__(self, num_features, num_hidden_units, num_layers, t, dropout, lin_layers):
        super().__init__()
        self.num_features = num_features
        self.hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.t = t

        self.lstm = nn.LSTM(
            num_features=num_features,
            hidden_size=num_hidden_units,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
            num_layers=self.num_layers
        )
        self.fc = nn.Linear(num_hidden_units, lin_layers)
        self.fc1 = nn.Linear(lin_layers, lin_layers)
        self.fc2 = nn.Linear(lin_layers, t)
        self.relu = nn.ReLU()

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        output = output[:, -1, :]

        # fully connected layers
        out = self.relu(output)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


def train_and_test_model(config, t=None, epochs=None, features=None, target=None, device=None, data_normaliser=None):
    sorted_grouped_job_files_dir = config["sorted_grouped_job_files_dir"]
    training_files, validation_files, test_files_csv = get_train_validation_test_csv_files(sorted_grouped_job_files_dir,
                                                                                           data_normaliser)
    training_loaders, validation_loaders = get_train_validation_data_loaders(training_files, validation_files, t,
                                                                             target, features, config, data_normaliser)
    model = RegressionLSTM(num_features
                           =len(features), num_hidden_units=config["units"],
                           num_layers=config["layers"], t=t, dropout=config["dropout"],
                           lin_layers=config["lin_layers"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    smallest_loss = float('inf')
    rounds_without_update = 0
    for epoch_index in range(epochs):
        for training_loader in training_loaders:
            train_model(training_loader, model, optimizer=optimizer, device=device, t=t)
        validation_mse = [test_model(validation_loader, model, device=device, t=t)
                          for validation_loader in validation_loaders]
        total_mse = sum(validation_mse)
        metrics = {"loss": total_mse,
                   "training_iteration": epoch_index}
        train.report(metrics)

        if smallest_loss > total_mse:
            smallest_loss = total_mse
            os.makedirs("my_model", exist_ok=True)
            torch.save(
                (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
            checkpoint = Checkpoint.from_directory("my_model")
            train.report(metrics, checkpoint=checkpoint)
            rounds_without_update = 0
        else:
            rounds_without_update += 1

        if (epoch_index > epochs / 4) and rounds_without_update >= config["stop_early_round_threshold"]:
            return


def create_config(sequence_length=12):
    config = {
        "sequence_length": sequence_length,
        "units": tune.grid_search([128]),
        "layers": tune.grid_search([1]),
        "lin_layers": tune.grid_search([500]),
        "lr": tune.grid_search([0.001]),
        "batch_size": tune.grid_search([32]),
        "sorted_grouped_job_files_dir": "/home/anna/PycharmProjects/time-series-optimisation/data/sortedGroupedJobFiles/",
        "stop_early_round_threshold": 10,
        "dropout": 0.2
    }
    return config


def train_with_tune(train_and_test_model, t, epochs, features, target, device, config,
                    num_samples=4, data_normaliser=None):
    result = tune.run(
        partial(train_and_test_model, t=t,
                epochs=epochs, features=features,
                target=target, device=device, data_normaliser=data_normaliser),
        resources_per_trial={"cpu": 4, "gpu": 0},
        config=config,
        num_samples=num_samples,
        scheduler=AsyncHyperBandScheduler(time_attr="training_iteration", metric="loss", mode="min", max_t=epochs,
                                          grace_period=epochs / 4, reduction_factor=2),
        progress_reporter=CLIReporter(metric_columns=["loss", "training_iteration"]),
    )
    return result


def append_best_trail_to_file(file_path, best_trial):
    append_to_file(file_path,
                   "u=" + str(best_trial.config["units"]) + ", l=" + str(best_trial.config["layers"]) + ", lr=" + str(
                       round(best_trial.config["lr"], 5)) + ", bs=" +
                   str(best_trial.config["batch_size"]) + ", ll=" +
                   str(best_trial.config["lin_layers"]))


def get_data_sequences(files_csv, t, target, features, config, data_normaliser):
    data_sequences = []
    for data_file in files_csv:
        data_sequence = get_test_data(t, target, features, data_file, config, data_normaliser)
        data_sequences.append(data_sequence)
    return data_sequences


def calculate_and_print_results(t, predictions, actuals, start_time, training_time, label):
    print(f"Calculating {label} results")
    calculate_prediction_results(t, predictions, actuals, start_time, training_time, f"new_data_{label}")


def get_results(t, target, test_data, best_trained_model, device, best_trial_config, training_data, validation_data,
                start_time, training_time, data_normaliser):
    print("Get test results")
    pred_test, act_test = get_prediction_results(t, target, test_data, best_trained_model, device,
                                                 best_trial_config, data_normaliser)
    print("Get training results")
    pred_train, act_train = get_prediction_results(t, target, training_data, best_trained_model, device,
                                                   best_trial_config, data_normaliser)
    print("Get validation results")
    pred_validation, act_validation = get_prediction_results(t, target, validation_data, best_trained_model,
                                                             device, best_trial_config, data_normaliser)

    print("calculate results")
    calculate_prediction_results(t, pred_train, act_train, start_time, training_time,
                                 "../results/lstm_results_cpu_NEW/train")
    calculate_prediction_results(t, pred_test, act_test, start_time, training_time,
                                 "../results/lstm_results_cpu_NEW/test")
    calculate_prediction_results(t, pred_validation, act_validation, start_time, training_time,
                                 "../results/lstm_results_cpu_NEW/validation")


def main(t=1, sequence_length=12, epochs=2000, features=['mean_CPU_usage'], target=["mean_CPU_usage"],
         num_samples=100):
    set_seed()
    file_path = './lstm_NEW.txt'
    append_to_file(file_path, "t=" + str(t) + ", sequence length=" + str(sequence_length) + ", epochs=" + str(epochs))
    start_time = time.time()
    config = create_config(sequence_length)
    data_normaliser = DataNormaliser()
    training_files_csv, validation_files_csv, test_files_csv = get_train_validation_test_csv_files(
        config["sorted_grouped_job_files_dir"], data_normaliser)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    result = train_with_tune(train_and_test_model, t, epochs, features,
                             target, device, config, num_samples=num_samples, data_normaliser=data_normaliser)
    for trial in result.trials:
        print(trial.metric_analysis)
    best_trial = result.get_best_trial("loss", "min", "last")
    training_time = round((time.time() - start_time), 2)
    append_to_file(file_path, "training_time=" + str(training_time))

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    append_to_file(file_path, str(best_trial.checkpoint))
    best_trained_model = RegressionLSTM(num_features=len(features), num_hidden_units=best_trial.config["units"],
                                        num_layers=best_trial.config["layers"], t=t, dropout=0,
                                        lin_layers=best_trial.config["lin_layers"])
    best_trained_model.to(device)
    print(best_trial.checkpoint)
    checkpoint_path = os.path.join(best_trial.checkpoint.to_directory(), "checkpoint.pt")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    training_data = get_data_sequences(training_files_csv, t, target, features, best_trial.config, data_normaliser)
    test_data = get_data_sequences(test_files_csv, t, target, features, best_trial.config, data_normaliser)
    validation_data = get_data_sequences(validation_files_csv, t, target, features, best_trial.config, data_normaliser)

    get_results(t, target, test_data, best_trained_model, device, best_trial.config, training_data, validation_data,
                start_time, training_time, data_normaliser)


if __name__ == "__main__":
    main(t=6, sequence_length=6, epochs=1,
         # features=['start_time', 'mean_CPU_usage', 'canonical_mem_usage', 'assigned_mem_usage',
         #           'unmapped_page_cache_mem_usage', 'total_page_cache_mem_usage', 'max_mem_usage',
         #           'mean_disk_IO_time', 'mean_local_disk_space_used', 'max_CPU_usage', 'max_disk_IO_time', 'CPI', 'MAI',
         #           'sampled_CPU_usage', 'nr_of_tasks', 'scheduling_class'],
         features=['mean_CPU_usage'], target=['mean_CPU_usage'], num_samples=1)
