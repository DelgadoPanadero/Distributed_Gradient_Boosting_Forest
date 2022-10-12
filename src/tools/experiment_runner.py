import os
import json

from inspect import importlib
from importlib import import_module

import numpy as np
import pandas as pd


class ExperimentRunner():


    def __init__(self, config):


        try:
            self._load_models(config)

        except:
            raise Exception('Error loading model modules')

        try:
            self._load_datasets(config)
        except:
            raise Exception('Error loading datatset')

        try:
            self._load_experiments(config)
        except:
            raise Exception('Error loading experiment modules')


    @property
    def models(self):
        return self._models


    @property
    def datasets(self):
        return self._datasets


    @property
    def experiments(self):
        return self._experiments


    def _load_models(self, config):

        models = {}
        for name, model_config in config['Models'].items():

            module = model_config['module']
            object = model_config['object']
            params = model_config['parameters']

            models[name] = getattr(import_module(module), object)(**params)

        self._models = models

        return models


    def _load_experiments(self, config):

        experiments = {}
        for experiment in config['Experiments']:

            module = experiment['module']
            object = experiment['object']
            params = experiment['parameters']

            params.update(self._models)

            experiments[object] = getattr(import_module(module),object)(**params)

        self._experiments = experiments

        return experiments


    def _load_datasets(self, config):

        datasets = {}
        for dataset in config['Datasets']:

            if not os.path.exists(dataset['file']):

                os.mkdir('data', exist_ok=True)
                func = getattr(pd, dataset['function'])

                try:
                    df = func(dataset['url'], sep=dataset['sep'])
                except KeyError:
                    df = func(dataset['url'])

                df.to_csv(dataset['file'], index=False)

            data = pd.read_csv(dataset['file'])
            X, y = np.array(data.iloc[:, 0:-1]), np.array(data.iloc[:, -1])

            datasets[dataset['name']] = (X, y)

        self._datasets = datasets

        return datasets


    def run(self):

        for dataset_name, dataset in self.datasets.items():
            X,y = dataset
            for name, experiment in self.experiments.items():
                experiment.run(dataset_name,X,y)
