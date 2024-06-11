import os
from datetime import timedelta, timezone

import pandas as pd

from utils.sequence_data import create_sliding_window


def get_train_validation_test_csv_files(sorted_grouped_job_files_dir, data_normaliser):
    file_list = os.listdir(sorted_grouped_job_files_dir)
    file_list.sort()
    sorted_file_list = file_list
    num_files = len(sorted_file_list)

    train_ratio = 0.6
    test_ratio = 0.2

    train_index = int(train_ratio * num_files)
    test_index = train_index + int(test_ratio * num_files)

    train_files = sorted_file_list[:train_index]
    test_files = sorted_file_list[train_index:test_index]
    validation_files = sorted_file_list[test_index:]
    training_files_csv = read_files(train_files, True, sorted_grouped_job_files_dir, data_normaliser)
    validation_files_csv = read_files(validation_files, False, sorted_grouped_job_files_dir, data_normaliser)
    test_files_csv = read_files(test_files, False, sorted_grouped_job_files_dir, data_normaliser)

    return training_files_csv, validation_files_csv, test_files_csv


def get_test_csv_files(sorted_grouped_job_files_dir, data_normaliser):
    file_list = os.listdir(sorted_grouped_job_files_dir)
    file_list.sort()
    test_files = file_list
    test_files_csv = read_files(test_files, False, sorted_grouped_job_files_dir, data_normaliser)
    return test_files_csv

def get_train_validation_test_x_y(t, sequence_length, features, target, sorted_grouped_job_files_dir):
    file_list = os.listdir(sorted_grouped_job_files_dir)
    file_list.sort()
    sorted_file_list = file_list
    num_files = len(sorted_file_list)

    train_ratio = 0.6
    test_ratio = 0.2

    train_index = int(train_ratio * num_files)
    test_index = train_index + int(test_ratio * num_files)

    train_files = sorted_file_list[:train_index]
    test_files = sorted_file_list[train_index:test_index]
    validation_files = sorted_file_list[test_index:]

    train_files_x, train_files_y = get_x_y(train_files, t, sequence_length, features, target,
                                           sorted_grouped_job_files_dir)
    validation_files_x, validation_files_y = get_x_y(validation_files, t, sequence_length, features, target,
                                                     sorted_grouped_job_files_dir)
    test_files_x, test_files_y = get_x_y(test_files, t, sequence_length, features, target, sorted_grouped_job_files_dir)

    return (train_files_x, train_files_y), (validation_files_x, validation_files_y), (test_files_x, test_files_y)


def get_test_x_y(t, sequence_length, features, target, sorted_grouped_job_files_dir):
    file_list = os.listdir(sorted_grouped_job_files_dir)
    file_list.sort()
    sorted_file_list = file_list
    test_files = sorted_file_list

    test_files_x, test_files_y = get_x_y(test_files, t, sequence_length, features, target, sorted_grouped_job_files_dir)

    return test_files_x, test_files_y


def get_x_y(files, t, sequence_length, features, target, sorted_grouped_job_files_dir):
    files_csv_x = list()
    files_csv_y = list()
    for file in files:
        df = pd.read_csv(sorted_grouped_job_files_dir + "/" + file, sep=",")
        df.index = pd.DatetimeIndex(df["start_time"])
        df.index = df.index.tz_localize(timezone.utc).tz_convert('US/Eastern')
        first_timestamp = df.index[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
        increment = timedelta(minutes=5)
        df.index = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
                    [first_timestamp + i * increment for i in range(len(df))]]
        X_train, y_train = create_sliding_window(t, sequence_length, df[features], df[target])
        files_csv_x.append(X_train)
        files_csv_y.append(y_train)

    return files_csv_x, files_csv_y


def append_to_file(file_path, content):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    try:
        with open(file_path, 'a+') as file:
            file.write(content)
            file.write('\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")


def read_files(files, is_training, dir_path, data_normalizer):
    files_csv = list()
    for file in files:
        df = pd.read_csv(dir_path + file, sep=",")
        if (is_training):
            data_normalizer.get_min_max_values_of_training_data(df)
        files_csv.append(df)
    return files_csv
