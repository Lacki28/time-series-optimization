import pandas as pd

import torch
class DataNormaliser:
    def __init__(self):
        self.min_max_dict = {}

    def normalize_data_minMax(self, features, df):
        pd.options.mode.chained_assignment = None
        # find min and max
        for c in df.columns:
            if c in features:
                min = self.min_max_dict[c]['min']
                max = self.min_max_dict[c]['max']
                value_range = max - min
                df.loc[:, c] = (df.loc[:, c] - min) / value_range
        return df

    def denormalize_data_minMax(self, target, prediction_test):
        prediction_test = (prediction_test * (
                self.min_max_dict[target]["max"] - self.min_max_dict[target]["min"])) + self.min_max_dict[
                              target]["min"]
        return prediction_test

    def get_min_max_values_of_training_data(self, df):
        for col in df.columns:
            if col not in self.min_max_dict:
                self.min_max_dict[col] = {"min": df[col].min(), "max": df[col].max()}
            else:
                self.min_max_dict[col]["min"] = min(self.min_max_dict[col]["min"], df[col].min())
                self.min_max_dict[col]["max"] = max(self.min_max_dict[col]["max"], df[col].max())
    def get_min_max_dict(self):
        return self.min_max_dict


