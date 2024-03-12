#!/usr/bin/env python3
"""Contains the code for ICAPAI'21 paper "Counterfactual Explanations for Multivariate Time Series"

Authors:
    Emre Ates (1), Burak Aksar (1), Vitus J. Leung (2), Ayse K. Coskun (1)
Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.
"""

from pathlib import Path
import os
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_dataset(set_name, binary=False, window_size = 10, exclude_columns=[], **kwargs):
    kwargs['window_size'] = window_size
    kwargs['windowize'] = False
    kwargs['skip'] = 1
    kwargs['trim'] = 1
    # kwargs['noise_scale'] = 1
    # kwargs['test_split_size'] = 1
    return get_dataset_csv(set_name, exclude_columns, **kwargs)

def drop_columns(timeseries):
    return timeseries.drop(
        [x for x in timeseries.columns
         if x.endswith('HASW') or 'per_core' in x], axis=1)


def select_classes(timeseries, labels, classes):
    if classes is None:
        return timeseries, labels
    labels = labels[labels['label'].isin(classes)]
    timeseries = timeseries.loc[labels.index.get_level_values('node_id'), :, :]
    return timeseries, labels


def process_data(timeseries, labels, classes=None, **kwargs):
    timeseries = drop_columns(timeseries)
    timeseries, labels = select_classes(timeseries, labels, classes=classes)
    timeseries = timeseries.dropna(axis=0)
    assert(not timeseries.isnull().any().any())
    return timeseries, labels

def get_dataset_csv(set_name, exclude_columns, window_size, **kwargs):
    pattern = kwargs.get('pattern', "*.csv")
    root_dir = kwargs.get('rootdir', './data/' + set_name)
    train_data, train_labels, test_data, test_labels = preprocess_data(root_dir, pattern, exclude_columns, window_size, **kwargs)
    return train_data, train_labels, test_data, test_labels

def add_noise(ts, noise_scale):

    if noise_scale == 0:
        return ts
    
    noisy_df = ts.copy()
    for column in ts.columns:
        if column not in ['Timestamp', 'label']:
            # Add noise to non-excluded columns
            std_dev = ts[column].mean()
            noise = np.random.normal(0, std_dev*noise_scale, ts.shape[0])  # Example of Gaussian noise
            noisy_df[column] += noise
    return noisy_df

def windowize(ts_list, window_size, node_offset = 0):
    windowized_ts = []

    for node_number in range(len(ts_list)):
        ts = ts_list[node_number]
        ts_window_temp = []
        node_number = node_number + node_offset

        for i in range(window_size, len(ts)):
            ts_window_temp = ts.iloc[i - window_size:i].copy()
            new_node_id = 'node_{}_{}'.format(node_number+1,i)
            ts_window_temp.index = pd.MultiIndex.from_product([[new_node_id], ts_window_temp['Timestamp']], names=['node_id', 'timestamp'])
            windowized_ts.append(ts_window_temp)

    windowized_df = pd.concat(windowized_ts)
    return windowized_df

def preprocess_data(root_dir, pattern="*.csv",  exclude_columns = [], window_size=5, noise_scale=0, use_classes=None,
                  **kwargs):
    
    data = {'train': [], 'test': []}

    for file_path in glob(os.path.join(root_dir, pattern)):
        
        file_name = os.path.basename(file_path)
        if 'train' in file_name:
            data_type = 'train'
        elif 'test' in file_name:
            data_type = 'test'
        else:
            continue

        ts = pd.read_csv(file_path)
        ts.drop(columns=exclude_columns, inplace=True)
        ts = add_noise(ts, noise_scale)

        data[data_type].append(ts)

    train_df = windowize(data['train'], window_size, 100)
    test_df = windowize(data['test'], window_size, 0)

    train_labels_df = train_df.groupby(level='node_id').agg({'label': 'last'})
    train_labels_df.index = pd.MultiIndex.from_product([train_labels_df.index, [0]], names=['node_id', 'timestamp'])
    test_labels_df = test_df.groupby(level='node_id').agg({'label': 'last'})
    test_labels_df.index = pd.MultiIndex.from_product([test_labels_df.index, [0]], names=['node_id', 'timestamp'])

    train_df.drop(columns=['Timestamp', 'label'], inplace=True)
    test_df.drop(columns=['Timestamp', 'label'], inplace=True)

    train_ts, train_labels = process_data(train_df, train_labels_df, use_classes)
    test_ts, test_labels = process_data(test_df, test_labels_df, use_classes)

    return train_ts, train_labels, test_ts, test_labels
