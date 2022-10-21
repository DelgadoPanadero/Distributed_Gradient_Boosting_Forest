# Boosted forest

This repo contains the soruce code of the Boosted Forest algorithm. The pretrained model artifact has not been upload because the training process is very quickly. 



## Install dependencies

```
$ pip install -r requirements.txt
```

## Experiments Execution

The  execution of this project performs a set of experiments to compare the metric results of `BoostedForest` with `GradientBoosting` in 200 experiments over a set of 4 datasets. The experiment configuration can be found in the file `config.json`

```
$ python run_experiments.py
```
The results of the experiments are save in the directory `results/`


## Quick Start

You can run the model over another dataset as follows

```python
from sklearn.datasets import load_wine

X,y=load_wine(return_X_y=True)

model = BoostedForest(n_layers=10, n_trees=10)
model.fit(X[0:-5,:],y[0:-5])

model.predict(X[-5:,:])
```

```
array([1.60187112, 1.61052181, 1.51189302, 1.44837233, 1.63621279])
```


## Data

Data available for experiments can be found in `data/data_config.json`. There is the information needed to read the data for the experiments. If the data files are not available locally, the config file also holds the needed information to download and store them.

### Parkinson

- [URL](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- Observations: 5875
- Features: 23

### Wine

- [URL](https://archive.ics.uci.edu/ml/datasets/wine)
- Observations: 1599
- Features: 11

### Concrete

- [URL](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
- Observations: 1030
- Features: 8

### NavalVessel

- [URL](http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants)
- Observations: 11934
- Features: 16

### Cargo2000

- [URL](https://archive.ics.uci.edu/ml/datasets/Cargo+2000+Freight+Tracking+and+Tracing)
- Observations: 3942
- Features: 98

### BikeSales

- [URL](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand)
- Observations: 8760
- Features: 14


### Temperature

- [URL](https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast)
- Observations: 7750
- Features: 25

### Superconduct

- [URL](https://archive.ics.uci.edu/ml/datasets/superconductivty+data)
- Observations: 21263
- Features: 81

### Obesity

- [URL](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+)
- Observations: 2112
- Features: 17
 
