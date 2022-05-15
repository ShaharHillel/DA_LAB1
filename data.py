import random
import pandas as pd
import os
from tqdm import tqdm
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import statsmodels.api as sm


def interpolate_data(patient_df):
    cols = patient_df.columns
    patient_df[cols] = patient_df[cols].interpolate(limit_direction='both', axis=0)
    return patient_df


def cut_first_rows(patient_df, amount_of_rows=30, end_idx=None):
    idx = patient_df.shape[0] if end_idx is None else end_idx
    return patient_df.iloc[max(idx - amount_of_rows, 0):idx]


def trend_and_features(patient_df, amount_of_rows=10, is_test=False):
    if is_test:
        end_idx = patient_df.shape[0]
        patient_df = patient_df.drop(['Age', 'Gender'], axis=1)  # remove fixed values and label because useless
    else:
        label = patient_df.SepsisLabel.max()
        end_idx = patient_df.SepsisLabel.argmax() + 1 if (label == 1) else patient_df.shape[0]
        patient_df = patient_df.drop(['Age', 'Gender', 'SepsisLabel'], axis=1)  # remove fixed values and label because useless

    trend = np.vstack([sm.tsa.seasonal_decompose(patient_df.loc[:, [col]], model='additive', period=1,
                                                 extrapolate_trend=1).trend for col in patient_df.columns]).T
    patient_values = cut_first_rows(patient_df, amount_of_rows, end_idx=end_idx).values
    trend = cut_first_rows(pd.DataFrame(trend), amount_of_rows, end_idx=end_idx).values

    mean, std, median = patient_values.mean(axis=0), patient_values.std(axis=0), np.median(patient_values, axis=0)
    t_min, t_max = trend.min(axis=0), trend.max(axis=0)
    first, last, diff, last_diff = trend[0], trend[-1], trend[-1] - trend[0], trend[-1] - trend[-2 if len(trend) >= 2 else -1]
    avgmin1 = sum([trend[i+1]-1 for i, tr in enumerate(trend[:-1])])/len(trend) if len(trend) > 1 else diff
    if is_test:
        return np.vstack((first, last, diff, mean, std, t_max, t_min, last_diff, avgmin1)).flatten('F')
    else:
        return np.vstack((first, last, diff, mean, std, t_max, t_min, last_diff, avgmin1)).flatten('F'), label


def process_data():
    random.seed(10)
    np.random.seed(10)
    os.makedirs('cache/data', exist_ok=True)
    all_data_path = f'cache/data/data_features.pkl'
    imputer_path = f'cache/data/imputer.pkl'
    if os.path.isfile(all_data_path):  
        return pickle.load(open(all_data_path, 'rb'))

    train_path_dir = "/home/student/dap_lab1/data/train/"
    test_path_dir = "/home/student/dap_lab1/data/test/"
    train_files = sorted([train_path_dir + file_name for file_name in os.listdir(train_path_dir)])
    test_files = sorted([test_path_dir + file_name for file_name in os.listdir(test_path_dir)])

    patients_dfs = []
    for path_file in tqdm(os.listdir(train_path_dir)):
        patient_df = pd.read_csv(train_path_dir + path_file, sep='|').drop(['Unit1', 'Unit2', 'ICULOS', 'SepsisLabel'], axis=1)
        patient_df = interpolate_data(patient_df)
        patients_dfs.append(patient_df)
    all_patients = pd.concat(patients_dfs)

    imputer = IterativeImputer(random_state=0)
    imputer.fit(all_patients.values)

    all_data = {}

    for mode, file_paths in tqdm(zip(['train', 'test'], [train_files, test_files])):
        features_list, labels_list = [], []
        for file_path in tqdm(file_paths):
            patient_df = pd.read_csv(file_path, sep='|')
            sepsis_data = patient_df['SepsisLabel'].values
            patient_df = patient_df.drop(['SepsisLabel', 'Unit1', 'Unit2', 'ICULOS'], axis=1)

            # drop those with almost no data
            if mode == 'train' and patient_df.count().sum() / (patient_df.shape[0] * patient_df.shape[1]) < 0.1:
                continue

            values = imputer.transform(interpolate_data(patient_df).values)
            patient_df = pd.DataFrame(values, columns=patient_df.columns)
            patient_df['SepsisLabel'] = sepsis_data

            # extract features
            features, label = trend_and_features(patient_df, amount_of_rows=30)
            features_list.append(features)
            labels_list.append(label)

        all_data[mode] = (np.array(features_list), np.array(labels_list))
        pickle.dump(all_data, open(all_data_path, 'wb'))
        pickle.dump(imputer, open(imputer_path, 'wb'))
    return all_data


def save_feature_names():
    train_path_dir = "/home/student/dap_lab1/data/train/"
    train_files = sorted([train_path_dir + file_name for file_name in os.listdir(train_path_dir)])

    patients_dfs = []
    for path_file in os.listdir(train_path_dir):
        patient_df = pd.read_csv(train_path_dir + path_file, sep='|')
        patient_df = patient_df.drop(['Unit1', 'Unit2', 'ICULOS'] + ['Age', 'Gender', 'SepsisLabel'], axis=1)
        feature_stats = ['first', 'last', 'diff', 'mean', 'std', 't_max', 't_min', 'last_diff', 'avgmin1']
        feature_names = np.array(sum([[f'{col}_{stat}' for stat in feature_stats] for col in patient_df.columns], []))

        pickle.dump(feature_names, open(f'cache/data/feature_names.pkl', 'wb'))
        return feature_names

