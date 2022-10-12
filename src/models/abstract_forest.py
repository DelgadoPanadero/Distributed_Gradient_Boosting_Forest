import copy
import itertools
from abc import ABC, abstractmethod

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class AbstractForestRegressor(ABC):


    """
    Template for all the models of the project
    """

    def __init__(
        self,
        n_feature,
        n_layers,
        max_depth,
        parallel_fit_jobs=None,
        parallel_predict_jobs=None):


        self._parallel_predict_jobs = parallel_predict_jobs
        self._parallel_fit_jobs     = parallel_fit_jobs
        self.n_features             = n_features
        self.max_depth              = max_depth
        self.n_layers               = n_layers

        self._graph                 = []
        self._aggregator            = []
        self._is_fitted             = False


    @property
    def is_fitted(self):

        """
        Atribute to check if the model is fittedt or not. IMPORTANT this
        atribute might be used by the fit() method during its execution
        to call predict() for fitting porpuse
        """

        return self._is_fitted


    @property
    def graph(self):

        """
        A list of list. This variable holds all the estimators of the ensemble.
        Each element of the list represents a Layer which is a list of tree,
        which represents each node of the layer.
        """

        return self._graph


    @property
    def _base_model(self):

        """
        The dummy estimator for the ensemble
        """

        return copy.deepcopy(DecisionTreeRegressor(max_depth=self.max_depth))


    #@abstractmethod()
    def fit(self,X,y):

        """
        Command method to choose the training function
        """

        raise NotImplementerError()


    #@abstractmethod()
    def predict(self,X):

        """
        Command method to choose the predict function
        """

        raise NotImplementerError()


    def _get_estimator_impurities(self,tree,X):

        """
        Returns the impurity of the leaf node for each sample
        """

        leaf_nodes = tree.apply(X)
        impurities = tree.tree_.impurity[leaf_nodes]

        return impurities


    def _fit_stage(self, X, residuals):

        """
        For the current fitting layer, fit every tree of it.
        """

        new_layer = []
        for set in self._feature_sets:
            new_layer.append(self._base_model.fit(X[:,set],residuals))

        return new_layer

    def _fit_stage_parallel(self, X, residuals):

        new_layer = [self._base_model for set in self._feature_sets]

        #Fit the model for every posible combination
        Parallel(n_jobs=self._parallel_fit_jobs, prefer="threads")(
                 delayed(self._parallel_fit_trees)(tree,
                                                   X,
                                                   residuals,
                                                   set)
                 for set, tree in zip(self._feature_sets, new_layer)
                 )

        return new_layer


    def _parallel_fit_trees(tree, X, y, set):
        tree.fit(X[:,set],y)
        return None


    def _parallel_predict(tree, X, idx, feature_sets):

        prediction = tree.predict(X[:,feature_sets[idx]])

        if len(prediction.shape) > 1:
            prediction = prediction[:,idx]

        leaf_nodes = tree.apply(X[:,feature_sets[idx]])
        impurities = tree.tree_.impurity[leaf_nodes]

        return (np.array(prediction), np.array(impurities))
