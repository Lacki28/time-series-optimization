import torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader

from utils.sequence_data import SequenceDataset
from utils.train_test_predict import predict


def set_seed(seed=28):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_test_data(t, target, features, df_test=None, config=None, data_normalizer=None):
    df_test = data_normalizer.normalize_data_minMax(features, df_test)
    test_sequence = SequenceDataset(
        df_test,
        target=target,
        features=features,
        t=t,
        sequence_length=config["sequence_length"])

    return test_sequence


def apply_savgol_filter_to_column(column):
    return savgol_filter(column, 51, 4)


def get_training_data(t, target, features, df_train=None, config=None, data_normalizer=None):
    df_train = df_train.apply(lambda x: apply_savgol_filter_to_column(x))
    df_train = data_normalizer.normalize_data_minMax(features, df_train)
    train_sequence = SequenceDataset(
        df_train,
        target=target,
        features=features, t=t,
        sequence_length=config["sequence_length"])

    return train_sequence


def get_train_validation_data_loaders(training_files, validation_files, t, target, features, config, data_normaliser):
    training_loaders = [
        DataLoader(get_training_data(t, target, features, df_train, config, data_normaliser),
                   batch_size=config["batch_size"],
                   shuffle=False) for df_train in training_files]
    validation_loaders = [
        DataLoader(get_test_data(t, target, features, df_validation, config, data_normaliser),
                   batch_size=config["batch_size"],
                   shuffle=False) for df_validation in validation_files]
    return training_loaders, validation_loaders


def get_prediction_results(t, target, test_data, best_trained_model, device, config, data_normalizer):
    pred_denorm_cpus = list()
    act_denorm_cpus = list()
    for data in test_data:
        print("h")
        test_eval_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        prediction_test_cpu = predict(test_eval_loader, best_trained_model, device)
        denorm_cpu = data_normalizer.denormalize_data_minMax(target[0], prediction_test_cpu)
        actual_test_cpu = data.y[:, 0]
        actual_test_cpu = actual_test_cpu.unfold(0, t, 1)
        act_denorm_cpu = data_normalizer.denormalize_data_minMax(target[0], actual_test_cpu)

        pred_denorm_cpus.append(denorm_cpu)
        act_denorm_cpus.append(act_denorm_cpu)
    return pred_denorm_cpus, act_denorm_cpus



