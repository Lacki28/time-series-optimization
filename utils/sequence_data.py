import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, t, sequence_length=5):
        self.features = features
        self.target = target
        self.t = t
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.t

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i+1: i + self.t+1]  # return target n time stamps ahead


def create_sliding_window(t, sequence_length, x_data, y_data):
    X = []
    y = []
    for i in range(0, len(x_data) - sequence_length):
        if i < sequence_length:
            padding = np.tile(np.array([x_data.iloc[0]]), sequence_length - i - 1).flatten()
            x_window = x_data.iloc[0:i + 1].values.flatten()
            x_window = np.concatenate((padding, x_window), axis=0)
        else:
            x_window = x_data.iloc[i - sequence_length + 1:i + 1].values.flatten()

        y_window = y_data.iloc[i + 1:i + t + 1].values.flatten()
        X.append(x_window)
        y.append(y_window)

    X = np.array(X)
    y = np.array(y)
    return X, y