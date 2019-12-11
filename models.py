#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.pipeline import Pipeline


LR_L2 = Pipeline([
    ('normalizer', preprocessing.Normalizer()),
    ('reg', linear_model.Ridge())
])

LR_L2 = Pipeline([
    ('normalizer', preprocessing.Normalizer()),
    ('reg', linear_model.Lasso())
])

SGD_REG = Pipeline([
    ('normalizer', preprocessing.Normalizer()),
    ('reg', linear_model.SGDRegressor())
])
SGD_params = {
    "loss": ["squared_loss", "huber"],
    "penalty": ["l2", "l1"],
    "alpha": [0.0001, 0.001, 1, 10],
    "shuffle": [True, False],
    "n_iter": [50]
}

RF = Pipeline([
    ('normalizer', preprocessing.Normalizer()),
    ('reg', ensemble.RandomForestRegressor(n_jobs=2))
])
param_grid = {
    "n_estimators": [10],
    "n_jobs": [2]
}
