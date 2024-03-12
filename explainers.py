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

import logging
import numbers
import multiprocessing
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
# import six
# import sys
# sys.modules['sklearn.externals.six'] = six

import data_loading

class BaseExplanation:
    def __init__(self, clf, timeseries, labels, silent=True,
                 num_distractors=2, dont_stop=False,
                 threads=multiprocessing.cpu_count()):
        self.clf = clf
        self.timeseries = timeseries
        self.labels = labels
        self.silent = silent
        self.num_distractors = num_distractors
        self.metrics = self.clf.steps[0][1].column_names
        self.dont_stop = dont_stop
        self.window_size = len(timeseries.loc[
            timeseries.index.get_level_values('node_id')[0]])
        self.ts_min = np.repeat(timeseries.min().values, self.window_size)
        self.ts_max = np.repeat(timeseries.max().values, self.window_size)
        self.ts_std = np.repeat(timeseries.std().values, self.window_size)
        self.tree = None
        self.per_class_trees = None
        self.threads = threads

    def explain(self, x_test, **kwargs):
        raise NotImplementedError("Please don't use the base class directly")

    def _get_feature_names(self, clf, timeseries):
        if hasattr(self.clf.steps[1][1], 'transform'):
            return self.clf.steps[2][1].column_names
        else:
            window_size = len(timeseries.loc[
                [timeseries.index.get_level_values('node_id')[0]], :, :])
            names = []
            for c in timeseries.columns:
                for i in range(window_size):
                    names.append(c + '_' + str(i) + 's')
            return names

    def _transform_data(self, data, sample=None):
        if hasattr(self.clf.steps[1][1], 'transform'):
            transformed = self.clf.steps[1][1].transform(data)
            if sample:
                transformed = transformed.sample(sample)
            return self.clf.steps[3][1].transform(transformed)
        else:
            # autoencoder
            train_set = []
            for node_id in data.index.get_level_values('node_id').unique():
                train_set.append(data.loc[[node_id], :, :].values.T.flatten())
            result = np.stack(train_set)
            if sample:
                idx = np.random.randint(len(result), size=sample)
                result = result[idx, :]
            return result

    def _plot_changed(self, metric, original, distractor, savefig=False):
        fig = plt.figure(figsize=(6,3))
        ax = fig.gca()
        plt.plot(range(-distractor.shape[0], 0),
                 original[metric].values, label='x$_{test}$',
                 figure=fig,
                 )
        plt.plot(range(-distractor.shape[0], 0),
                 distractor[metric].values, label='Distractor',
                 figure=fig)
        ax.set_ylabel(metric)
        ax.set_xlabel('Time (s)')
        ax.legend()
        if savefig:
            filename = "{}.pdf".format(uuid.uuid4())
            fig.savefig(filename, bbox_inches='tight')
            logging.info("Saved the figure to %s", filename)
        fig.show()

    def _plot_changed2(self, metric, original, distractor, savefig=False):
        fig = plt.figure(figsize=(6,3))
        ax = fig.gca()
        plt.plot(range(1-distractor.shape[0], 1),
                 original[metric].values, label='Fault',
                 figure=fig)
        plt.plot(range(1, distractor.shape[0]+1),
                 distractor[metric].values, label='Normal',
                 figure=fig)
        plt.xticks(np.arange(1-distractor.shape[0], distractor.shape[0], step=1), rotation=45)
        ax.set_ylabel(metric)
        ax.set_xlabel('Time (s)')
        ax.legend()
        if savefig:
            filename = "{}.pdf".format(uuid.uuid4())
            fig.savefig(filename, bbox_inches='tight')
            logging.info("Saved the figure to %s", filename)
        fig.show()

    def construct_per_class_trees(self):
        """Used to choose distractors"""
        if self.per_class_trees is not None:
            return
        self.per_class_trees = {}
        self.per_class_node_indices = {c: [] for c in self.clf.classes_}
        preds = self.clf.predict(self.timeseries)
        true_positive_node_ids = {c: [] for c in self.clf.classes_}
        for pred, (idx, row) in zip(preds, self.labels.iterrows()):
            if row['label'] == pred:
                true_positive_node_ids[pred].append(idx[0])
        for c in self.clf.classes_:
            dataset = []
            for node_id in true_positive_node_ids[c]:
                dataset.append(self.timeseries.loc[
                    [node_id], :, :].values.T.flatten())
                self.per_class_node_indices[c].append(node_id)
            self.per_class_trees[c] = KDTree(np.stack(dataset))
        if not self.silent:
            logging.info("Finished constructing per class kdtree")

    def construct_tree(self):
        if self.tree is not None:
            return
        train_set = []
        self.node_indices = []
        for node_id in self.timeseries.index.get_level_values(
                'node_id').unique():
            train_set.append(self.timeseries.loc[
                [node_id], :, :].values.T.flatten())
            self.node_indices.append(node_id)
        self.tree = KDTree(np.stack(train_set))
        if not self.silent:
            logging.info("Finished constructing the kdtree")

    def _get_distractors(self, x_test, to_maximize, n_distractors=2):
        self.construct_per_class_trees()
        # to_maximize can be int, string or np.int64
        if isinstance(to_maximize, numbers.Integral):
            to_maximize = self.clf.classes_[to_maximize]
        distractors = []
        for idx in self.per_class_trees[to_maximize].query(
                x_test.values.T.flatten().reshape(1, -1),
                k=n_distractors)[1].flatten():
            distractors.append(self.timeseries.loc[
                [self.per_class_node_indices[to_maximize][idx]], :, :])
        if not self.silent:
            logging.info("Returning distractors %s", [
                x.index.get_level_values('node_id').unique().values[0]
                for x in distractors])
        return distractors

    def _get_recourse_distractors(self, x_test, to_maximize, n_distractors=2):
        # self.construct_per_class_trees()
        # to_maximize can be int, string or np.int64

        if isinstance(to_maximize, numbers.Integral):
            to_maximize = self.clf.classes_[to_maximize]
        
        min_dist = 100
        distractors = []

        for node, group in self.labels.groupby(level=['node_id','timestamp']):
            if group['label'][0] == to_maximize:
                distractor = self.timeseries.loc[(node[0], slice(None)),:]
                distance = np.linalg.norm(distractor.iloc[0] - x_test.iloc[-1])
                if(distance < min_dist):
                    min_dist = distance
                    min_dist_node = node[0]
                    logging.info("Min dist:%s Min dist node:%s", min_dist, min_dist_node)
        distractor = self.timeseries.loc[(min_dist_node, slice(None)),:]
        distractors.append(distractor)

        if not self.silent:
            logging.info("Returning distractors %s", [
                x.index.get_level_values('node_id').unique().values[0]
                for x in distractors])
        for column in x_test.columns:
            self._plot_changed2(column, x_test, distractor, savefig=False)
        return distractors

CLASSIFIER = None
X_TEST = None
DISTRACTOR = None

def _eval_one(tup):
    column, label_idx = tup
    global CLASSIFIER
    global X_TEST
    global DISTRACTOR
    x_test = X_TEST.copy()
    x_test[column] = DISTRACTOR[column].values
    return CLASSIFIER.predict_proba(x_test)[0][label_idx]


class BruteForceSearch(BaseExplanation):
    def _find_best(self, x_test, distractor, label_idx):
        global CLASSIFIER
        global X_TEST
        global DISTRACTOR
        CLASSIFIER = self.clf
        X_TEST = x_test
        DISTRACTOR = distractor
        best_case = self.clf.predict_proba(x_test)[0][label_idx]
        best_column = None
        tuples = []
        for c in distractor.columns:
            if np.any(distractor[c].values != x_test[c].values):
                tuples.append((c, label_idx))
        if self.threads == 1:
            results = []
            for t in tuples:
                results.append(_eval_one(t))
        else:
            pool = multiprocessing.Pool(self.threads)
            results = pool.map(_eval_one, tuples)
            pool.close()
            pool.join()
        for (c, _), pred in zip(tuples, results):
            if pred > best_case:
                best_column = c
                best_case = pred
        if not self.silent:
            logging.info("Best column: %s, best case: %s",
                         best_column, best_case)
        return best_column, best_case


    def get_prototype(self, x_test, to_maximize=None, num_features=10,return_dist=False, savefig=False):
        orig_preds = self.clf.predict_proba(x_test)
        orig_label = np.argmax(orig_preds)
        if to_maximize is None:
            to_maximize = np.argmin(orig_preds)
        if orig_label == to_maximize:
            return []
        if not self.silent:
            logging.info("Working on turning label from %s to %s",
                         orig_label, to_maximize)
        distractors = self._get_recourse_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors)
                
        new_ts = []
        new_ts.append(x_test)
        new_ts.append(distractors[0])
        new_df = pd.concat(new_ts)
        start_time = x_test.index.get_level_values('timestamp')[0]
        new_df['Timestamp'] = range(start_time, start_time + new_df.shape[0])
        # new_df.set_index(['Timestamp'], inplace=True)
        new_df = new_df.reset_index()
        new_df.drop(columns=['node_id',  'timestamp'], inplace=True)
        # new_df.index = pd.MultiIndex.from_product(
        #     [[x_test.index.get_level_values('node_id').values[0]], new_df['t']], names=['node_id', 'timestamp'])
        new_df['label'] = 0

        new_ts_set = []
        new_ts_set.append(new_df)
        proto_df = data_loading.windowize(new_ts_set, self.window_size)
        # proto_df.drop(columns=['Timestamp'], inplace=True)

        proto_labels_df = proto_df.groupby(level='node_id').agg({'label': 'last'})
        proto_labels_df.index = pd.MultiIndex.from_product([proto_labels_df.index, [0]], names=['node_id', 'timestamp'])

        proto_df.drop(columns=['Timestamp', 'label'], inplace=True)

        proto_ts, proto_labels = data_loading.process_data(proto_df, proto_labels_df)
        # self._pred_dist(distractor)
        return proto_ts, proto_labels

    def explain(self, x_test, to_maximize=None, num_features=10,return_dist=False, savefig=False):
        orig_preds = self.clf.predict_proba(x_test)
        orig_label = np.argmax(orig_preds)
        if to_maximize is None:
            to_maximize = np.argmin(orig_preds)
        if orig_label == to_maximize:
            return []
        if not self.silent:
            logging.info("Working on turning label from %s to %s",
                         orig_label, to_maximize)
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors)
        best_explanation = set()
        best_explanation_score = 0
        for count, dist in enumerate(distractors):
            if not self.silent:
                logging.info("Trying distractor %d / %d",
                             count + 1, self.num_distractors)
            explanation = []
            modified = x_test.copy()
            prev_best = 0
            best_dist = dist #TODO: Only supports one distractor
            while True:
                probas = self.clf.predict_proba(modified)
                if not self.silent:
                    logging.info("Current probas: %s", probas)
                if np.argmax(probas) == to_maximize:
                    current_best = np.max(probas)
                    if current_best > best_explanation_score:
                        best_explanation = explanation
                        best_explanation_score = current_best
                    if current_best <= prev_best: # stop when proba starts decreasing
                        break
                    prev_best = current_best
                    if not self.dont_stop:
                        break
                if (not self.dont_stop and
                        len(best_explanation) != 0 and
                        len(explanation) >= len(best_explanation)):
                    break
                best_column, _ = self._find_best(modified, dist, to_maximize) # find best column to increase pred proba
                if best_column is None:
                    break
                if not self.silent:
                    self._plot_changed(best_column, modified, dist, savefig=savefig)
                modified[best_column] = dist[best_column].values # update column in modified
                explanation.append(best_column)

        if not return_dist:
            return explanation
        return explanation, best_dist
