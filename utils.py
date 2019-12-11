#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate_mse(model, x_train, x_dev, y_train, y_dev):
    """ """
    # mse on training set
    y_pred_train = model.predict(x_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    print("MSE on training set: %.3f" % mse_train)
    # mse on test set
    y_pred_val = model.predict(x_dev)
    mse_val = mean_squared_error(y_dev, y_pred_val)
    print("MSE on test set: %.3f" % mse_val)
    #
    return mse_train, mse_val


def build_test_fold(train, dev):
    """ """
    a = pd.DataFrame(-1, index=train.index, columns=["fold"])
    b = pd.DataFrame(0, index=dev.index, columns=["fold"])
    test_fold = pd.concat([a, b], axis=0)
    return test_fold


def get_feature_importances(df, model):
    """Plot feature importance like xgb.plot_importances()"""
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    features = df.columns
    indices = np.argsort(importances)
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(np.arange(0, len(indices) * 2, 2), importances[indices],
             height=1, color='b', align='center')
    plt.yticks(np.arange(0, len(indices) * 2, 2), features[indices])
    plt.margins(y=0)
    plt.show()


def shuffle_XY(X, Y):
    """ """
    tmp = pd.concat([X, Y], axis=1).sample(frac=1)
    return tmp[X.columns], tmp[Y.columns]
