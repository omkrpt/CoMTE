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

import sys
import logging
import random
from pathlib import Path
import os
from glob import glob
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_dataset(set_name, binary=False, window_size = 10, exclude_columns=[], **kwargs):
    if set_name in ['cstr', 'pronto', 'cstr2']:
        kwargs['window_size'] = window_size
        kwargs['windowize'] = False
        kwargs['skip'] = 1
        kwargs['trim'] = 1
        # kwargs['noise_scale'] = 1
        # kwargs['test_split_size'] = 1
        return get_dataset_csv(set_name, exclude_columns, **kwargs)
    if set_name not in ['taxonomist', 'hpas', 'test', 'natops']:
        raise ValueError("Wrong set_name")
    if binary:
        if set_name in ['taxonomist', 'test', 'natops']:
            kwargs['make_binary'] = True
        elif set_name == 'hpas':
            kwargs['classes'] = ['none', 'dcopy']
    rootdir = Path(kwargs.get('rootdir', './data'))
    if set_name == 'taxonomist':
        kwargs['window'] = 45
        kwargs['skip'] = 45
    if set_name == 'test':
        set_name = 'taxonomist'
        kwargs['window'] = 60
        kwargs['skip'] = 60
        kwargs['test'] = True
    if set_name == 'natops':
        kwargs['windowize'] = False
    return load_hpc_data(rootdir / set_name, **kwargs)

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


def load_hpc_data(data_folder, make_binary=False, for_autoencoder=False, **kwargs):
    if for_autoencoder:
        # Only get data from a single hardware node
        if 'none' not in kwargs.get('classes'):
            raise ValueError("Autoencoder has to train with healthy class")
        nodeid_df = pd.read_csv(data_folder / 'nids.csv')
        labels = pd.concat([pd.read_hdf(data_folder / 'train_labels.hdf'),
                            pd.read_hdf(data_folder / 'test_labels.hdf')])
        labels = labels[labels['label'].isin(kwargs.get('classes'))]
        best_nid = 0
        best_count = 0
        for nid in nodeid_df['nid'].unique():
            node_ids = nodeid_df[nodeid_df['nid'] == nid]['node_id']
            if len(labels.loc[node_ids, :, :]['label'].unique()) == 1:
                continue
            min_count = labels.loc[node_ids, :, :]['label'].value_counts().min()
            if min_count > best_count:
                best_nid = nid
                best_count = min_count

        node_ids = nodeid_df[nodeid_df['nid'] == best_nid]['node_id']
        labels = labels.loc[node_ids, :, :]
        logging.info("Returning runs from nid000%d, counts: %s",
                     best_nid, labels['label'].value_counts().to_dict())
        timeseries = pd.concat([pd.read_hdf(data_folder / 'train.hdf'),
                                pd.read_hdf(data_folder / 'test.hdf')])

        train_nodeids, test_nodeids = train_test_split(
            labels.index.get_level_values('node_id').unique(), test_size=0.2, random_state=0)
        test_timeseries = timeseries.loc[test_nodeids, :, :]
        test_labels = labels.loc[test_nodeids, :, :]
        timeseries = timeseries.loc[train_nodeids, :, :]
        labels = labels.loc[train_nodeids, :, :]
    else:

        timeseries = pd.read_hdf(data_folder / 'train.hdf')
        labels = pd.read_hdf(data_folder / 'train_labels.hdf')
        labels['label'] = labels['label'].astype(str)
        if make_binary:
            label_to_keep = labels.mode()['label'][0]
            labels[labels['label'] != label_to_keep] = 'other'

        test_timeseries = pd.read_hdf(data_folder / 'test.hdf')
        test_labels = pd.read_hdf(data_folder / 'test_labels.hdf')
        test_labels['label'] = test_labels['label'].astype(str)
        if make_binary:
            test_labels[test_labels['label'] != label_to_keep] = 'other'

    timeseries, labels = process_data(timeseries, labels, **kwargs)
    assert(not timeseries.isnull().any().any())
    test_timeseries, test_labels = process_data(
        test_timeseries, test_labels, **kwargs)
    assert(not test_timeseries.isnull().any().any())

    # Divide test data
    if kwargs.get('test', False):
        test_node_ids = [
            test_labels[test_labels['label'] == label].index.get_level_values('node_id')[0]
            for label in test_labels['label'].unique()]
        test_labels = test_labels.loc[test_node_ids, :]
        test_timeseries = test_timeseries.loc[test_node_ids, :]

    return timeseries, labels, test_timeseries, test_labels

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

def windowize(ts_df, window_size, node_offset = 0):
    windowized_ts = []

    for node_number in range(len(ts_df)):
        ts = ts_df[node_number]
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

    train_df = windowize_new(data['train'], window_size, 100)
    test_df = windowize_new(data['test'], window_size, 0)

    train_labels_df = train_df.groupby(level='node_id').agg({'label': 'last'})
    train_labels_df.index = pd.MultiIndex.from_product([train_labels_df.index, [0]], names=['node_id', 'timestamp'])
    test_labels_df = test_df.groupby(level='node_id').agg({'label': 'last'})
    test_labels_df.index = pd.MultiIndex.from_product([test_labels_df.index, [0]], names=['node_id', 'timestamp'])

    train_df.drop(columns=['Timestamp', 'label'], inplace=True)
    test_df.drop(columns=['Timestamp', 'label'], inplace=True)

    train_ts, train_labels = process_data(train_df, train_labels_df, use_classes)
    test_ts, test_labels = process_data(test_df, test_labels_df, use_classes)

    return train_ts, train_labels, test_ts, test_labels
