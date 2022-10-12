import abc

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class AbstractModelTest():


    @abc.abstractmethod
    def create_batch(self, X, y):
        raise NotImplementedError()


    @abc.abstractmethod
    def run(self, name, X, y):
        raise NotImplementedError()


    @abc.abstractmethod
    def score(self, y_test, y_pred):
        raise NotImplementedError()


    def _save_summary(self, name, scores_1, scores_2, scores_3):

        differences_1 = scores_3-scores_1
        differences_2 = scores_3-scores_2

        file_name = f'results/{name.replace(" ","_").lower()}.txt'
        with open(file_name, "w") as file:
            text = f'Model 1 Score: {scores_1.mean()} +- {scores_1.std()}\n'+\
                   f'Model 2 Score: {scores_2.mean()} +- {scores_2.std()}\n'+\
                   f'Model 3 Score: {scores_3.mean()} +- {scores_3.std()}\n'+\
                   f'Differences: {differences_1.mean()} +- {differences_1.std()}'+\
                   f'Differences: {differences_2.mean()} +- {differences_2.std()}'
            file.write(text)


    def _save_scores(self, name, scores_1, scores_2, score_3):

        file_name = f'results/{name.replace(" ","_").lower()}_scores.csv'
        with open(file_name, "w") as file:
            text = 'model_1, model_2, model_3, differences_1, differences_2\n'
            for score_1,score_2 in zip(scores_1,scores_2):
                text+=f'{score_1}, {score_2}, {score_3}, {score_3-score_1}, {score_3-score_2}\n'
            file.write(text)


    def _plot_scores_diff(self, name, scores_1, scores_2, scores_3, n_bins):

        differences = scores_3-scores_1
        top = differences.max()
        bottom = differences.min()
        bins = np.linspace(bottom, top, n_bins)
        plt.hist(differences, bins, alpha=0.5, label='R2 diff GBDT')

        differences = scores_3-scores_2
        top = differences.max()
        bottom = differences.min()
        bins = np.linspace(bottom, top, n_bins)
        plt.hist(differences, bins, alpha=0.5, label='R2 diff RF')

        plt.title(f'{name} Paired R2 Difference')
        plt.xlabel("Paired R2 Diference")
        plt.ylabel("Counts")
        plt.legend(loc='upper right')
        file_name = f'results/{name.replace(" ","_").lower()}_score_diff.png'
        plt.savefig(file_name)
        plt.clf()


    def _plot_scores(self, name, scores_1, scores_2, scores_3, n_bins):

        top = max(scores_1.max(),scores_2.max(), scores_3.max())
        bottom = min(scores_1.min(),scores_2.min(), scores_3.min())
        bins = np.linspace(bottom, top, n_bins)

        plt.hist(scores_1, bins, alpha=0.5, label='GBDT')
        plt.hist(scores_2, bins, alpha=0.5, label='RF')
        plt.hist(scores_3, bins, alpha=0.5, label='DGBF')

        plt.title(f'{name} R2 Score')
        plt.xlabel("R2 Score")
        plt.ylabel("Counts")
        plt.legend(loc='upper right')
        file_name = f'results/{name.replace(" ","_").lower()}_score.png'
        plt.savefig(file_name)
        plt.clf()


    def _plot_preds(self, name, y_test, y_pred1, y_pred2, n_bins):

        top = max(y_test.max(), y_pred1.max(),y_pred2.max())
        bottom = min(y_test.min(), y_pred1.min(),y_pred2.min())
        bins = np.linspace(bottom, top, n_bins)
        plt.hist(y_test, bins, alpha=0.5, label='test')
        plt.hist(y_pred1, bins, alpha=0.5, label='GB')
        plt.hist(y_pred2, bins, alpha=0.5, label='BF')
        plt.title(f'{name} Preds')
        plt.xlabel("Prediction values")
        plt.ylabel("Counts")
        plt.legend(loc='upper right')
        file_name = f'results/{name.replace(" ","_").lower()}_preds.png'
        plt.savefig(file_name)
        plt.clf()


    def _plot_deviations(self, name, y_test, y_pred1, y_pred2, n_bins):

        diff1 = y_pred1-y_test
        diff2 = y_pred2-y_test
        top = max(diff1.max(), diff2.max())
        bottom = min(diff1.min(), diff2.min())
        bins = np.linspace(bottom, top, n_bins)
        plt.hist(diff1, bins, alpha=0.5, label='GB')
        plt.hist(diff2, bins, alpha=0.5, label='BF')
        plt.title(f'{name} Deviations')
        plt.xlabel("Prediction deviation")
        plt.ylabel("Counts")
        plt.legend(loc='upper right')
        file_name = f'results/{name.replace(" ","_").lower()}_deviations.png'
        plt.savefig(file_name)
        plt.clf()
