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
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import cPickle as pickle
from utils import shuffle_XY


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

features_config = {
    "NO2": {
        "rolling_mean": True,
        # "deltas_mean": [6, 48, 96, 120, 192],
        # "rolling_std": True,
        # "deltas_std": [24, 48, 96, 120],
        "temp_dec_freq": 12,
        "pollutant": False,
        "roll_mean_conf": roll_mean_conf,
        "shift_config": shift_config,
        "remove_temporal": True
    },
    "PM10": {
        "rolling_mean": True,
        # "deltas_mean": [6, 48, 96, 120, 192],
        # "rolling_std": True,
        # "deltas_std": [24, 48, 96],
        "temp_dec_freq": 48,
        "pollutant": True,
        "roll_mean_conf": roll_mean_conf,
        "shift_config": shift_config,
        "remove_temporal": True
    },
    "PM25": {
        "rolling_mean": True,
        # "deltas_mean": [6, 48, 96, 120, 192],
        # "rolling_std": True,
        # "deltas_std": [24, 48, 96],
        "temp_dec_freq": 48,
        "pollutant": True,
        "roll_mean_conf": roll_mean_conf,
        "shift_config": shift_config,
        "remove_temporal": True
    },
    "PM": {
        "rolling_mean": True,
        # "deltas_mean": [6, 48, 96, 120, 192],
        # "rolling_std": True,
        # "deltas_std": [24, 48, 96],
        "temp_dec_freq": 48,
        "pollutant": True,
        "roll_mean_conf": roll_mean_conf,
        "shift_config": shift_config,
        "remove_temporal": True
    },

}

rf_config = {
    "n_estimators": 30,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None,
    "min_impurity_split": 1e-07,
    "n_jobs": 4
}

xgb_config = {
    "NO2": {
        "n_estimators": 300,
        "reg_lambda": 1
    },
    "PM10": {
        "n_estimators": 300,
        "reg_lambda": 1
    },
    "PM25": {
        "n_estimators": 250,
        "reg_lambda": 1
    }
}


def train(dataset, labels):
    """ """
    pollutants = ["NO2", "PM10", "PM25"]
    # split dataset
    NO2_df, PM10_df, PM25_df = split_pollutant_dataset(dataset)
    # build data dict
    ds = dict(((poll, df) for poll, df in zip(pollutants, split_pollutant_dataset(dataset))))
    # build features dict
    f = {}
    for poll in pollutants:
        f[poll] = {}
        f[poll]["X"] = make_features(ds[poll], **features_config[poll])
        f[poll]["Y"] = get_Y(labels, ds[poll])
    # train model for each pollutant
    model_dict = {}
    for poll in pollutants:
        xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=200,
                                     reg_lambda=1)
        # train model
        xgb_model.fit(f[poll]["X"], f[poll]["Y"])
        # mse on training set
        y_pred = xgb_model.predict(f[poll]["X"])
        mse = mean_squared_error(f[poll]["Y"], y_pred)
        print("%s: MSE on training set: %.3f" % (poll, mse))
        # store model
        model_dict[poll] = xgb_model
    # return model dict
    return model_dict


def predict(model_dict, dataset):
    """ """
    # split dataset
    NO2_df, PM10_df, PM25_df = split_pollutant_dataset(dataset)
    # build features
    NO2_f = make_features(NO2_df, **features_config["NO2"])
    PM10_f = make_features(PM10_df, **features_config["PM10"])
    PM25_f = make_features(PM25_df, **features_config["PM25"])
    # apply each model
    Y_pred_NO2 = pd.DataFrame(model_dict["NO2"].predict(NO2_f),
                              columns=["TARGET"], index=NO2_f.index)
    Y_pred_PM10 = pd.DataFrame(model_dict["PM10"].predict(PM10_f),
                               columns=["TARGET"], index=PM10_f.index)
    Y_pred_PM25 = pd.DataFrame(model_dict["PM25"].predict(PM25_f),
                               columns=["TARGET"], index=PM25_f.index)
    # concatenate result
    Y_pred = pd.concat([Y_pred_NO2, Y_pred_PM10, Y_pred_PM25], axis=0)
    # return
    return Y_pred


def train_predict(train, test, Y_train, model_dict=None, output_path=None,
                  pm=False, model="rf"):
    """ """
    pollutants = ["NO2", "PM"] if pm else ["NO2", "PM10", "PM25"]
    print("%i regressor will be trained for each pollutant of %s" %
          (len(pollutants), pollutants))
    # split dataset, build data dict
    train_ds = dict(((poll, df) for poll, df in
                    zip(pollutants, split_pollutant_dataset(train, pm))))
    test_ds = dict(((poll, df) for poll, df in
                    zip(pollutants, split_pollutant_dataset(test, pm))))
    # build features dict
    f = {}
    for poll in pollutants:
        f[poll] = {}
        f[poll]["X_train"], f[poll]["X_test"] = make_features(
            train_ds[poll], dev=test_ds[poll], **features_config[poll])
        if Y_train is not None:
            f[poll]["Y"] = get_Y(Y_train, train_ds[poll])
    # train model for each pollutant
    if model_dict is None:
        model_dict = {}
        for poll in pollutants:
            # shuffle X,Y
            X, Y = shuffle_XY(f[poll]["X_train"], f[poll]["Y"])
            # init model
            if model == "rf":
                reg = RandomForestRegressor(**rf_config)
            else:
                reg = xgb.XGBRegressor(max_depth=6, **xgb_config[poll])
            # train model
            print("Training a %s model on pollutant %s ..." % (model, poll))
            reg.fit(X, Y)
            print("Training done on %s" % poll)
            # store model
            model_dict[poll] = reg
        if output_path is not None:
            print("Saving the dictionnary of models in %s" % output_path)
            with open(output_path, "wb") as fout:
                pickle.dump(model_dict, fout)
    # predict on train set
    preds = []
    for poll in pollutants:
        # mse on training set
        Y_pred_poll = pd.DataFrame(
            model_dict[poll].predict(f[poll]["X_train"]),
            columns=["TARGET"],
            index=f[poll]["X_train"].index)
        preds.append(Y_pred_poll)
        mse = mean_squared_error(f[poll]["Y"], Y_pred_poll)
        print("%s: MSE on training set: %.3f" % (poll, mse))
    # concat and compute global MSE
    Y_pred = pd.concat(preds, axis=0).sort_index()
    mse = mean_squared_error(Y_train, Y_pred)
    print("GLOBAL MSE on training set: %.3f" % mse)
    # predict on test set
    print("Computing prediction on test data...")
    preds = []
    for poll in pollutants:
        Y_pred_poll = pd.DataFrame(
            model_dict[poll].predict(f[poll]["X_test"]),
            columns=["TARGET"],
            index=f[poll]["X_test"].index)
        preds.append(Y_pred_poll)
    # concatenate pred for each pollutant and sort index
    Y_pred = pd.concat(preds, axis=0).sort_index()
    print("Prediction done.")
    #
    return Y_pred


if __name__ == '__main__':
    # Paths
    X_train_path = "C:/Users/Victor/Documents/MVA/Data Challenge/Plume/X_train.csv"
    X_test_path = "C:/Users/Victor/Documents/MVA/Data Challenge/Plume/X_test.csv"
    Y_train_path = "C:/Users/Victor/Documents/MVA/Data Challenge/Plume/Y_train.csv"
    # prepare data / features
    X_train = pd.read_csv(X_train_path, index_col="ID")
    X_train = preprocess_dataset(X_train)
    Y_train = pd.read_csv(Y_train_path, index_col="ID")
    X_test = pd.read_csv(X_test_path, index_col="ID")
    X_test = preprocess_dataset(X_test)
    # train
    # model_dict = train(X_train, Y_train)
    # # save models
    # pickle.dump(model_dict, open("model/model_dict_3.pkl", 'wb'))
    # # predict
    # Y_pred = predict(model_dict, X_test)
    y_pred = train_predict(X_train, X_test, Y_train,

    output_path="model/model_dict_rf.pkl",
                           pm=True)
    y_pred.to_csv("pred_rf.csv") 

    # with open("model/model_dict_3.pkl", "rb") as f:
    #     model_dict = pickle.load(f)


