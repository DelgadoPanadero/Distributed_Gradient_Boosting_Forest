import copy
import random
import itertools
import numpy as np
from scipy.optimize import minimize

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from .abstract_forest import AbstractForestRegressor


class BoostedForest(AbstractForestRegressor):


    def __init__(
        self,
        n_trees=10,
        n_layers=10,
        max_depth=None,
        learning_rate=0.1):

        self.n_trees       = n_trees
        self.n_layers      = n_layers
        self.max_depth     = max_depth
        self.learning_rate = learning_rate


    def _bootstrap_sample(self, X, pseudo_y, l, t):

        """
        """

        # Sample Size
        pos  = (X.shape[0]//self.n_trees)
        size = pos + ((X.shape[0]-pos)//(self.n_layers)) * (l+1)

        # Bootstrap Sample
        samples_ids = np.random.choice(
            np.arange(X.shape[0]),
            size=(size,))

        return X[samples_ids,:], pseudo_y[samples_ids,]


    def _compute_tree_output_weights(self, tree_pred, y_real):

        """
        """

        # Weigths optimization function
        def optimization_function(weights,tree_pred, y_real):
            y_pred = (tree_pred*weights).sum(axis=1)
            return mean_squared_error(y_real, y_pred, squared=False)

        # Initial guess with uniform distribution (aritmetic mean)
        weights_init = np.array([1/self.n_trees for _ in range(self.n_trees)])

        # Weights must be between 0 and 1
        weights_bounds = [(0.0,1.0) for _ in range(self.n_trees)]

        # Weights must sum 1 (to be average)
        weights_constraint = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1})

        # Optimization process to find the weights
        opt = minimize(
            fun=optimization_function,
            x0=weights_init,
            args=(tree_pred, y_real),
            bounds=weights_bounds,
            constraints=weights_constraint
            )

        weights = opt.x


        return weights


    def _fit_stage(self, X, pseudo_y_o, l):

        """
        For the current fitting layer, fit every tree of it.
        """

        new_layer = []
        new_weights = []

        for t in range(self.n_trees):

            # Train Test split
            X_lt, pseudo_y_lto = self._bootstrap_sample(X, pseudo_y_o, l, t)

            # Fit estimator
            tree = self._base_model.fit(X_lt,pseudo_y_lto)

            # Compute weights from layer l, tree t and outputs o
            w_lto = self._compute_tree_output_weights(
                tree_pred = tree.predict(X).reshape((X.shape[0],-1)),
                y_real    = pseudo_y_o.mean(axis=1)
            )

            new_weights.append(w_lto)
            new_layer.append(tree)

        self._weights.append(new_weights)

        return new_layer


    def _predict_stage(self, X):

        """
        """

        pred = np.full((X.shape[0],self.n_trees), self._prior)
        for l, layer in enumerate(self._graph[0:-1]):

            layer_pred = np.zeros((X.shape[0],self.n_trees))
            for t, tree in enumerate(layer):
                tree_pred_o = tree.predict(X).reshape((X.shape[0],-1))
                tree_pred = (tree_pred_o*self._weights[l][t]).sum(axis=1)
                layer_pred[:,t] += tree_pred

            pred += layer_pred.mean(axis=1).reshape((-1,1))

        for l, layer in enumerate(self._graph[-1:]):
            for t, tree in enumerate(layer):
                tree_pred_o = tree.predict(X).reshape((X.shape[0],-1))
                tree_pred = (tree_pred_o*self._weights[l][t]).sum(axis=1)
                pred[:,t] += tree_pred

        return pred


    def fit(self,X,y):

        """
        """

        # Init variables
        self._graph       = []
        self._weights     = []
        self._prior = y.mean()

        for l in range(self.n_layers):

            # Compute predictiona
            y_pred_o = self._predict_stage(X)

            # Get residuals
            pseudo_y_o = y.reshape((-1,1))-y_pred_o

            # Learning rate
            pseudo_y_o = pseudo_y_o*self.learning_rate

            # New layer
            self._graph.append(self._fit_stage(X, pseudo_y_o, l))

        return self


    def predict(self,X):

        """
        Predict Function

        This predict function takes all the residual prediction of each tree
        and extracts the one associated with the residual of the previous
        layer estimator with the same feature_set samples.
        """

        return self._predict_stage(X).mean(axis=1)

