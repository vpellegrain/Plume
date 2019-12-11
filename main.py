#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataset import (
    split_pollutant_dataset, preprocess_dataset, split_train_dev,
    get_Y, zone_station_train, zone_station_dev
)
from features import (
    make_features, make_seqential_features, get_seq_Y
)
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn import preprocessing
from sklearn import ensemble

from sklearn.neural_network import MLPRegressor


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



X_train_path = "/Users/thomasopsomer/data/plume-data/X_train.csv"
X_test_path = "/Users/thomasopsomer/data/plume-data/X_test.csv"
Y_train_path = "/Users/thomasopsomer/data/plume-data/Y_train.csv"

# load all dataset
df = pd.read_csv(X_train_path, index_col="ID")
df = preprocess_dataset(df)
Y = pd.read_csv(Y_train_path, index_col="ID")


# split for each pollutant
NO2_df, PM10_df, PM25_df = split_pollutant_dataset(df)

# split in train / dev for each pollutant
NO2_train, NO2_dev = split_train_dev(NO2_df, zone_station_train, zone_station_dev)
PM10_train, PM10_dev = split_train_dev(PM10_df, zone_station_train, zone_station_dev)
PM25_train, PM25_dev = split_train_dev(PM25_df, zone_station_train, zone_station_dev)

# make features and get labels

# NO2
NO2_train_f, NO2_dev_f = make_features(NO2_train, NO2_dev, normalize=False,
                                       rolling_mean=True, deltas=[12])
Y_NO2_train = get_Y(Y, NO2_train)
Y_NO2_dev = get_Y(Y, NO2_dev)
X_NO2 = pd.concat([NO2_train_f, NO2_dev_f], axis=0, copy=False)
Y_NO2 = pd.concat([Y_NO2_train, Y_NO2_dev], axis=0, copy=False)
NO2_test_fold = build_test_fold(Y_NO2_train, Y_NO2_dev)

# PM10
PM10_train_f, PM10_dev_f = make_features(PM10_train, PM10_dev)
Y_PM10_train = get_Y(Y, PM10_train)
Y_PM10_dev = get_Y(Y, PM10_dev)
X_PM10 = pd.concat([PM10_train_f, PM10_dev_f], axis=0, copy=False)
Y_PM10 = pd.concat([Y_PM10_train, Y_PM10_dev], axis=0, copy=False)
PM10_test_fold = build_test_fold(Y_PM10_train, Y_PM10_dev)

# PM25
PM25_train_f, PM25_dev_f = make_features(PM25_train, PM25_dev)
Y_PM25_train = get_Y(Y, PM25_train)
Y_PM25_dev = get_Y(Y, PM25_dev)
X_PM25 = pd.concat([PM25_train_f, PM25_dev_f], axis=0, copy=False)
Y_PM25 = pd.concat([Y_PM25_train, Y_PM25_dev], axis=0, copy=False)
PM25_test_fold = build_test_fold(Y_PM25_train, Y_PM25_dev)


# Linear model:

# regression L2 (ridge)
lr = linear_model.Ridge(alpha=1, normalize=True)
lr.fit(NO2_train_f, Y_NO2_train)
evaluate_mse(lr, NO2_train_f, NO2_dev_f, Y_NO2_train, Y_NO2_dev)

ps = PredefinedSplit(NO2_test_fold)

lr = linear_model.Ridge()
param_grid = {
    "alpha": [0.1, 1, 10, 100, 1000, 10000],
    "normalize": [True]
}
gs = GridSearchCV(lr, param_grid, scoring="neg_mean_squared_error", n_jobs=1,
                  iid=False, refit=True, cv=ps)
gs.fit(X_NO2, Y_NO2)
gs.best_params_
gs.best_score_


# regression L1 (lasso)
lr = linear_model.Lasso()

ps = PredefinedSplit(PM25_test_fold)

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1],
    "normalize": [True]
}
gs = GridSearchCV(lr, param_grid, scoring="neg_mean_squared_error", n_jobs=1,
                  iid=False, refit=True, cv=ps)
gs.fit(X_PM25, Y_PM25)
gs.best_params_
gs.best_score_



# Support Vector Regression
from sklearn.svm import SVR, LinearSVR


ps = PredefinedSplit(NO2_test_fold)
param_grid = {
    # "kernel": ["rbf"],
    "C": [0.1, 1, 10],
    "epsilon": [0.1]
}
svr = LinearSVR(C=10)
gs = GridSearchCV(svr, param_grid, scoring="neg_mean_squared_error", n_jobs=1,
                  iid=False, refit=True, cv=ps)

svr.fit(preprocessing.normalize(X_NO2), Y_NO2)
gs.fit(preprocessing.normalize(X_NO2), Y_NO2)

evaluate_mse(
    svr,
    preprocessing.normalize(NO2_train_f),
    preprocessing.normalize(NO2_dev_f),
    Y_NO2_train, Y_NO2_dev)


# SGD Regressor
ps = PredefinedSplit(NO2_test_fold)
param_grid = {
    "loss": ["squared_loss", "huber"],
    # "penalty": ["l2", "l1"],
    "penalty": ["l2", "l1"],
    "alpha": [0.0001, 0.001, 1, 10],
    "shuffle": [True, False],
    "n_iter": [10]
}
sgd = linear_model.SGDRegressor()
gs = GridSearchCV(sgd, param_grid, scoring="neg_mean_squared_error", n_jobs=1,
                  iid=False, refit=True, cv=ps)

gs.fit(preprocessing.Normalizer().fit_transform(X_NO2), Y_NO2)


print gs.best_params_
print gs.best_score_

evaluate_mse(gs,
             preprocessing.Normalizer().fit_transform(NO2_train_f),
             preprocessing.Normalizer().fit_transform(NO2_dev_f),
             Y_NO2_train, Y_NO2_dev)


# Ensemble

ps = PredefinedSplit(NO2_test_fold)
param_grid = {
    "n_estimators": [10],
    "n_jobs": [2]
}
rfr = ensemble.RandomForestRegressor(n_jobs=2)
gs = GridSearchCV(rfr, param_grid, scoring="neg_mean_squared_error", n_jobs=1,
                  iid=False, refit=True, cv=ps)

gs.fit(preprocessing.normalize(X_NO2), Y_NO2)

rfr = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=2)

rfr.fit(preprocessing.normalize(PM10_train_f), Y_PM10_train)

evaluate_mse(
    rfr,
    preprocessing.normalize(PM10_train_f),
    preprocessing.normalize(PM10_dev_f),
    Y_PM10_train, Y_PM10_dev)


def feature_importance(model, cols):
    m = model.feature_importances_.max()
    r = zip(cols, model.feature_importances_)
    r = map(lambda x: (x[0], x[1] * 100 / m), r)
    return sorted(r, reverse=True, key=lambda x: x[1])


# NN

mlp = MLPRegressor(hidden_layer_sizes=(250, 150), activation='relu')
mlp.fit(preprocessing.normalize(X_NO2), Y_NO2)

evaluate_mse(
    mlp,
    preprocessing.normalize(NO2_train_f),
    preprocessing.normalize(NO2_dev_f),
    Y_NO2_train, Y_NO2_dev)








# test features:
import xgboost as xgb
from features import drop_cols
from dataset import cols


NO2_train_f, NO2_dev_f = make_features(
    NO2_train, NO2_dev, normalize=False,
    rolling_mean=True, deltas_mean=[24, 48, 96, 144, 288],
    temp_dec_freq=24)

NO2_train_f = drop_cols(NO2_train_f, cols["temporal"])
NO2_dev_f = drop_cols(NO2_dev_f, cols["temporal"])
Y_NO2_train = get_Y(Y, NO2_train)
Y_NO2_dev = get_Y(Y, NO2_dev)

xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=200, reg_lambda=10)

xgb_model.fit(NO2_train_f, Y_NO2_train,
              eval_set=[(NO2_dev_f, Y_NO2_dev)],
              eval_metric="rmse")

evaluate_mse(xgb_model, NO2_train_f, NO2_dev_f,
             Y_NO2_train, Y_NO2_dev)


import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model)
plt.show()



from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


def shuffle_XY(X, Y):
    """ """
    tmp = pd.concat([X,Y], axis=1).sample(frac=1)
    return tmp[X.columns], tmp[Y.columns]


Y_NO2 = get_Y(Y, NO2_df)
#NO2_train, NO2_dev, Y_NO2_train, Y_NO2_dev = train_test_split(NO2_df, Y_NO2)

NO2_train_f, NO2_dev_f = make_features(
    NO2_train, NO2_dev, normalize=False,
    rolling_mean=True, deltas_mean=[6, 48, 96, 120, 192],
    rolling_std=True, deltas_std=[24, 48, 96, 120],
    temp_dec_freq=12, log=False)

NO2_train_f = drop_cols(NO2_train_f, cols["temporal"])
NO2_dev_f = drop_cols(NO2_dev_f, cols["temporal"])
Y_NO2_train = get_Y(Y, NO2_train)
Y_NO2_dev = get_Y(Y, NO2_dev)

NO2_train_f, Y_NO2_train = shuffle_XY(NO2_train_f, Y_NO2_train)
NO2_dev_f, Y_NO2_dev = shuffle_XY(NO2_dev_f, Y_NO2_dev)

xgb_model = xgb.XGBRegressor(max_depth=9, n_estimators=100, reg_lambda=1)

xgb_model.fit(NO2_train_f, Y_NO2_train,
              eval_set=[(NO2_dev_f, Y_NO2_dev)],
              eval_metric="rmse")


PM25_train_f, PM25_dev_f = make_features(
    PM25_train, PM25_dev, normalize=False,
    rolling_mean=True, deltas_mean=[6, 48, 96, 120, 192],
    rolling_std=True, deltas_std=[24, 48, 96, 120],
    temp_dec_freq=48, log=False)
PM25_train_f = drop_cols(PM25_train_f, cols["temporal"])
PM25_dev_f = drop_cols(PM25_dev_f, cols["temporal"])
Y_PM25_train = get_Y(Y, PM25_train)
Y_PM25_dev = get_Y(Y, PM25_dev)

xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=300, reg_lambda=1)

xgb_model.fit(PM25_train_f, Y_PM25_train,
              eval_set=[(PM25_dev_f, Y_PM25_dev)],
              eval_metric="rmse")
evaluate_mse(xgb_model, PM25_train_f, PM25_dev_f,
             Y_PM25_train, Y_PM25_dev)

# Set a minimum threshold of 0.25
selection = SelectFromModel(xgb_model, threshold=0.01, prefit=True)
select_X_train = selection.transform(NO2_train_f)
select_X_dev = selection.transform(NO2_dev_f)

xgb_model_as = xgb.XGBRegressor(max_depth=6, n_estimators=200, reg_lambda=10)

xgb_model_as.fit(select_X_train, Y_NO2_train,
                 eval_set=[(select_X_dev, Y_NO2_dev)],
                 eval_metric="rmse")

evaluate_mse(xgb_model_as, select_X_train, select_X_dev,
             Y_NO2_train, Y_NO2_dev)

