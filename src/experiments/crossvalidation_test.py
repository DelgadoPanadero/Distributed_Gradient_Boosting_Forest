from math import ceil

import numpy as np
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from .abstract_test import AbstractModelTest


class CrossValidationModelTest(AbstractModelTest):


    def __init__(self,
                 model_1,
                 model_2,
                 n_runs=10,
                 n_bins=100,
                 n_folds=10,
                 test_size=0.25):

        self.model_1 = model_1
        self.model_2 = model_2
        self._n_runs = n_runs
        self._n_bins = n_bins
        self._n_folds = n_folds
        self._test_size = test_size


    def score(self, y_test, y_pred):
        return r2_score(y_test,y_pred)


    def create_batch(self, X, y):

        for i in range(self._n_folds):
            n_rows = X.shape[0]

            idx = list(range((i+0)*n_rows//self._n_folds,
                             (i+1)*n_rows//self._n_folds))
            filter = [False if i in idx else True for i in range(n_rows)]

            X_train = X[filter, :]
            y_train = y[filter   ]

            filter  = [True if i in idx else False for i in range(n_rows)]
            X_test  = X[filter, :]
            y_test  = y[filter   ]

            yield X_train, y_train, X_test, y_test


    def run(self, name, X, y):

        scores_1, scores_2 = [np.zeros((self._n_runs,))  for i in range(2)]
        y_tests, y_pred_1, y_pred_2 = [np.zeros(y.shape) for i in range(3)]

        for j in tqdm(range(self._n_runs)):

            ids = np.arange(y.shape[0])
            np.random.shuffle(ids)
            X = X[ids,:]
            y = y[ids]
            batch_iterator = self.create_batch(X, y)

            for i, batch_data in enumerate(batch_iterator):

                X_train, y_train, X_test, y_test = batch_data

                n_rows = X.shape[0]
                idx = list(range((i+0)*n_rows//self._n_folds,
                                 (i+1)*n_rows//self._n_folds))

                y_tests[ idx] += y_test/self._n_runs

                y_pred = self.model_1.fit(
                             X_train,
                             y_train
                         ).predict(X_test)
                y_pred_1[idx] += y_pred/self._n_runs
                scores_1[j] += self.score(y_test,y_pred)/self._n_folds

                y_pred = self.model_2.fit(
                             X_train,
                             y_train
                         ).predict(X_test)
                y_pred_2[idx] += y_pred/self._n_runs
                scores_2[j] += self.score(y_test,y_pred)/self._n_folds


        name = f'{name} Cross Validation Test'
        self.plot_results(name,y_tests, y_pred_1, y_pred_2, scores_1, scores_2)


    def plot_results(self,
                     name,
                     y_test,
                     y_pred_1,
                     y_pred_2,
                     scores_1,
                     scores_2):

        self._save_scores(     name, scores_1, scores_2)
        self._save_summary(    name, scores_1, scores_2)
        self._plot_scores(     name, scores_1, scores_2, self._n_folds)
        self._plot_scores_diff(name, scores_1, scores_2, self._n_folds)

        self._plot_preds(name, y_test, y_pred_1, y_pred_2, self._n_bins)
        self._plot_deviations(name, y_test, y_pred_1, y_pred_2, self._n_bins)
