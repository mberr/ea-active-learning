# Active Learning for Entity Alignment

[![Arxiv](https://img.shields.io/badge/arXiv-2001.08943-b31b1b)](https://arxiv.org/abs/2001.08943)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-2d618c?logo=python)](https://docs.python.org/3.8/)
[![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for the paper

```
Active Learning for Entity Alignment
Max Berrendorf*, Evgeniy Faerman*, and Volker Tresp
https://arxiv.org/abs/2001.08943
```

# Installation

Setup and activate virtual environment:

```shell script
python3.8 -m venv ./venv
source ./venv/bin/activate
```

Install requirements (in this virtual environment):

```shell script
pip install -U pip
pip install -U -r requirements.txt
```

# Preparation

In order to track results to a MLFlow server, start it first by running

```shell script
mlflow server
```

_Note: When storing the result for many configurations, we recommend to setup a database backend following the [instructions](https://mlflow.org/docs/latest/tracking.html)._
For the following examples, we assume that the server is running at

```shell script
TRACKING_URI=http://localhost:5000
```

# Experiments

For all experiments the results are logged to the running MLFlow instance. You can inspect the results during training by accessing the `TRACKING_URI` through a browser.

## Random Baseline

To run the random baseline use

```bash
cd executables
PYTHONPATH=../src python3 evaluate_active_learning_heuristic.py --phase=random --tracking_uri=${TRACKING_URI}
```

## Hyperparameter Search

To run the hyperparameter search use

```bash
cd executables
PYTHONPATH=../src python3 evaluate_active_learning_heuristic.py --phase=hpo --tracking_uri=${TRACKING_URI}
```

_Note: The hyperparameter searches takes a significant amount of time (~multiple days), and requires access to GPU(s). You can abort the script at any time, and inspect the current results via the web interface of MLFlow._

## Best Configurations

To rerun the best configurations we found in our hyperparameter search use

```bash
cd executables
PYTHONPATH=../src python3 evaluate_active_learning_heuristic.py --phase=best --tracking_uri=${TRACKING_URI}
```
