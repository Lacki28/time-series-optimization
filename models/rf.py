import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils.deep_learning_utils import set_seed
from utils.file_operations import get_train_validation_test_x_y
from utils.train_test_predict import predict_rf


def main(t=1, sequence_length=12, features=['mean_CPU_usage'], target=["mean_CPU_usage"],
         max_depth=8, trees=100):
    sorted_grouped_job_files_dir = "/home/anna/PycharmProjects/time-series-optimisation/data/sortedGroupedJobFilesmini/"
    set_seed()
    random_seed = 28
    start_time = time.time()
    (train_files_x, train_files_y), (validation_files_x, validation_files_y), (
        test_files_x, test_files_y) = get_train_validation_test_x_y(t, sequence_length, features, target,
                                                                    sorted_grouped_job_files_dir)

    regressor = RandomForestRegressor(n_estimators=trees, max_depth=max_depth, random_state=random_seed,
                                      criterion="squared_error")
    train_files_csv_x = np.concatenate(train_files_x, axis=0)  # Flatten the nested arrays
    train_files_csv_y = np.concatenate(train_files_y, axis=0)  # Flatten the nested arrays

    regressor.fit(train_files_csv_x, train_files_csv_y)

    joblib.dump(regressor, 'random_forest_' + str(sequence_length) + '_sequence_length_' + str(t) + '_max_depth_' + str(
        max_depth) + "_trees_" + str(trees) +str(target)+ '.pkl')
    training_time = round((time.time() - start_time), 2)
    hyperparams = "_" + str(max_depth) + "_" + str(trees)
    predict_rf(train_files_x, train_files_y, t, sequence_length, start_time, training_time, regressor,
               "../results/rf_results_mem" + hyperparams + "/train")
    print("predicted train")
    predict_rf(test_files_x, test_files_y, t, sequence_length, start_time, training_time, regressor,
               "../results/rf_results_mem" + hyperparams + "/test")
    print("predicted test")
    predict_rf(validation_files_x, validation_files_y, t, sequence_length, start_time, training_time, regressor,
               "../results/rf_results_mem" + hyperparams + "/validation")
    print("predicted validation")


# parameters tried: max depth 4, 8, 10, 16, 20, 32 (with 300 trees)
# Trees: 200, 300, 400
#Best model 300 trees, 32 max depth
if __name__ == "__main__":
    main(t=6, sequence_length=6, features=['canonical_mem_usage'], target=['canonical_mem_usage'], max_depth=4,
             trees=30)
