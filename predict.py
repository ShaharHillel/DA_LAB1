import sys
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from data import trend_and_features, interpolate_data
from train import print_metrics

from data import process_data

if __name__ == '__main__':
    test_path_dir = sys.argv[1]

    test_file_ids = sorted([f'{file_name[8:-4]}' for file_name in os.listdir(test_path_dir)])
    test_files = sorted([f'{test_path_dir}/{file_name}' for file_name in os.listdir(test_path_dir)])
    imputer_path = f'cache/data/imputer.pkl'
    imputer = pickle.load(open(imputer_path, 'rb'))


    features_list = []
    test_labels = []
    for file_path in tqdm(test_files):
        patient_df = pd.read_csv(file_path, sep='|')
        sepsis_data = patient_df['SepsisLabel'].values ###########
        if 'SepsisLabel' in list(patient_df.columns):
            test_labels.append(patient_df.SepsisLabel.max())
            patient_df = patient_df.drop(['SepsisLabel'], axis=1)
            mode = 'train'
        else:
            mode = 'test'
        patient_df = patient_df.drop(['Unit1', 'Unit2', 'ICULOS'], axis=1)

        values = imputer.transform(interpolate_data(patient_df).values)
        patient_df = pd.DataFrame(values, columns=patient_df.columns)
        patient_df['SepsisLabel'] = sepsis_data ##########
        # extract features
        features, _ = trend_and_features(patient_df, amount_of_rows=30, is_test=(mode == 'test'))
        features_list.append(features)

    features_array = np.array(features_list)
    clf = pickle.load(open("clf.pkl", 'rb'))

    probs = clf.predict_proba(features_array)
    preds = (probs[:, 1] >= 0.45).astype(int)

    print_metrics('test', clf, features_array, np.array(test_labels), prob_thresh=0.45)
    results = pd.DataFrame(data={'id': np.array(test_file_ids), 'preds': preds})
    results.to_csv("prediction.csv", header=False, index=False)


