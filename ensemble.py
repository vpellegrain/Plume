#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataset import (
    split_pollutant_dataset, preprocess_dataset, split_train_dev,
    get_Y, zone_station_train, zone_station_dev
)
from features import (
    make_features, make_seqential_features, get_seq_Y
)
from utils import build_test_fold, evaluate_mse
import pandas as pd

from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb



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


# feature conf
roll_mean_conf = {
    2: ["windspeed", "cloudcover"],
    4: ["windspeed", "cloudcover"],
    5: ["precipintensity", "precipprobability"],
    6: ["temperature"],
    10: ["precipintensity"],
    12: ["pressure", "cloudcover"],
    18: ["windbearingcos", "windbearingsin", "temperature"],
    24: ["pressure", "precipprobability", "windbearingcos"],
    32: ["windbearingsin"],
    48: ["pressure", "windbearingcos", "windbearingsin"],
    15: ["windspeed"],
    96: ["windbearingcos", "temperature", "windspeed"],
    144: ["temperature", "pressure"],
    288: ["temperature", "cloudcover"],
}

shift_config = {
    "temperature": [8, 14, 20, 96],
    "cloudcover": [2, 5, 48],
    "pressure": [2, 24, 72],
    "windbearingsin": [2, 6],
    "windbearingcos": [6, 6],
    "windspeed": [2, 4]
}

NO2_train, NO2_dev = split_train_dev(NO2_df, zone_station_train, zone_station_dev)


NO2_train_f, NO2_dev_f = make_features(
    NO2_train, NO2_dev,
    rolling_mean=True, roll_mean_conf=roll_mean_conf,
    # shift_config=shift_config,
    temp_dec_freq=12, log=False,
    remove_temporal=True,
    rolling_std=True,
    deltas_std=[24, 48, 96, 120])

Y_NO2_train = get_Y(Y, NO2_train)
Y_NO2_dev = get_Y(Y, NO2_dev)

# xgboost
xgb_model = xgb.XGBRegressor(max_depth=7, n_estimators=200, reg_lambda=1)

xgb_model.fit(NO2_train_f, Y_NO2_train,
              eval_set=[(NO2_dev_f, Y_NO2_dev)],
              eval_metric="rmse")

evaluate_mse(xgb_model, NO2_train_f, NO2_dev_f,
             Y_NO2_train, Y_NO2_dev)

# random forest
rf = RandomForestRegressor(
    n_estimators=10, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07,
    bootstrap=True, n_jobs=4)

NO2_train_f, Y_NO2_train = shuffle_XY(NO2_train_f, Y_NO2_train)
rf.fit(NO2_train_f, Y_NO2_train)

evaluate_mse(rf, NO2_train_f, NO2_dev_f, Y_NO2_train, Y_NO2_dev)




