from math import ceil
from tqdm import tqdm

import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .abstract_test import AbstractModelTest


class BootstrapModelTest(AbstractModelTest):


    def __init__(self,
                 model_1,
                 model_2,
                 model_3,
                 n_bins=100,
                 n_runs=100,
                 test_size=0.25):

        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self._n_runs = n_runs
        self._n_bins = n_bins
        self._test_size = test_size


    def score(self, y_test, y_pred):
        return r2_score(y_test,y_pred)


    def create_batch(self, X, y):

        for i in tqdm(range(self._n_runs)):

            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=self._test_size)

            yield X_train, y_train, X_test, y_test


    def run(self, name, X, y):

        batch_iterator = self.create_batch(X, y)

        scores_1, scores_2, scores_3 = [np.zeros((self._n_runs,)) for i in range(3)]
        y_tests, y_pred_1, y_pred_2, y_pred_3= [
            np.zeros((ceil(y.shape[0]*self._test_size),)) for i in range(4)]

        for i, batch_data in enumerate(batch_iterator):

            X_train, y_train, X_test, y_test = batch_data

            y_pred = self.model_1.fit(
                         X_train,
                         y_train
                     ).predict(X_test)
            scores_1[i] = self.score(y_test, y_pred)
            y_pred_1 += y_pred/self._n_runs

            y_pred = self.model_2.fit(
                         X_train,
                         y_train
                     ).predict(X_test)
            scores_2[i] = self.score(y_test, y_pred)
            y_pred_2 += y_pred/self._n_runs

            y_pred = self.model_3.fit(
                         X_train,
                         y_train
                     ).predict(X_test)
            scores_3[i] = self.score(y_test, y_pred)
            y_pred_3 += y_pred/self._n_runs

            y_tests  += y_test/self._n_runs

        name = f'{name} Bootstrap Test'
        self.plot_results(name, y_tests, y_pred_1, y_pred_2, y_pred_3,scores_1, scores_2, scores_3)


    def plot_results(self,
                     name,
                     y_test,
                     y_pred_1,
                     y_pred_2,
                     y_pred_3,
                     scores_1,
                     scores_2,
                     scores_3):

        self._save_scores(     name, scores_1, scores_2, scores_3)
        self._save_summary(    name, scores_1, scores_2, scores_3)
        self._plot_scores(     name, scores_1, scores_2, scores_3, self._n_bins)
        self._plot_scores_diff(name, scores_1, scores_2, scores_3, self._n_bins)

#        self._plot_preds(name, y_test, y_pred_1, y_pred_2,, y_pred_3, self._n_bins)
#        self._plot_deviations(name, y_test, y_pred_1, y_pred_2, y_pred_3, self._n_bins)

