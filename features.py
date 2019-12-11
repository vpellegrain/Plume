#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing and features extraction

# features ideas

for time:
    - keep daytime
    - add hour of day = (daytime - daytime(0)) % 24
    - add day of year = floor(k * (daytime - daytime(0)) / 24)

for each temporal information:
    - avg of temporal information for each zone
    - difference of temporal information to average

- station_id [1,2,3,4,5]
- integer (0/1) for calm_day or not

for each static features
preprocessing:


    - fill na with 0
    - scale data with MaxAbsScaler to handle sparse data
features:
    - a features stat indicate if the |feature| > 0 - is not NaN actually
    - the preprocessed value
"""
import pandas as pd
from sklearn import preprocessing
from dataset import cols
import numpy as np
import datetime


daytime_0 = 72.0


def fillna_static(df, log=False):
    """ """
    for col in cols["static"]:
        if log:
            df[col] = np.log(df.col.fillna(1.0))
        else:
            df[col] = df[col].fillna(0)


def scale_temporal(train, features_train, dev=None, features_dev=None):
    """ """
    robust_scaler = preprocessing.RobustScaler()
    tmp = pd.DataFrame(
        robust_scaler.fit_transform(train[cols["temporal"]]),
        columns=[col for col in cols["temporal"]],
        index=train.index)
    features_train = pd.concat([features_train, tmp], axis=1)
    if dev is not None:
        tmp = pd.DataFrame(
            robust_scaler.transform(dev[cols["temporal"]]),
            columns=[col for col in cols["temporal"]],
            index=dev.index)
        features_dev = pd.concat([features_dev, tmp], axis=1)
    return features_train, features_dev


def scale_static(train, features_train, dev=None, features_dev=None, log=False):
    """ """
    # fill static NaN with 0
    fillna_static(train, log)
    if dev is not None:
        fillna_static(dev, log)
    # scale data with MaxAbsScaler to handle sparse static data
    max_abs_scaler = preprocessing.MaxAbsScaler()
    tmp = pd.DataFrame(
        max_abs_scaler.fit_transform(train[cols["static"]]),
        columns=["%s_sc" % col for col in cols["static"]],
        index=train.index)
    features_train = pd.concat([features_train, tmp], axis=1)
    if dev is not None:
        tmp = pd.DataFrame(
            max_abs_scaler.transform(dev[cols["static"]]),
            columns=["%s_sc" % col for col in cols["static"]],
            index=dev.index)
        features_dev = pd.concat([features_dev, tmp], axis=1)
    return features_train, features_dev


def binarize_static(train, features_train, dev=None, features_dev=None):
    """ """
    # binary static features
    binarizer = preprocessing.Binarizer()
    tmp = pd.DataFrame(
        binarizer.fit_transform(train[cols["static"]]),
        columns=["%s_i" % col for col in cols["static"]],
        index=train.index)
    features_train = pd.concat([features_train, tmp], axis=1)
    if dev is not None:
        tmp = pd.DataFrame(
            binarizer.fit_transform(dev[cols["static"]]),
            columns=["%s_i" % col for col in cols["static"]],
            index=dev.index)
        features_dev = pd.concat([features_dev, tmp], axis=1)
    return features_train, features_dev


def delta_temporal_station_zone(train, features_train, dev=None,
                                features_dev=None):
    """ """
    # add difference of average in zone and value in station
    for col in cols["temporal"]:
        features_train["delta_%s" % col] = train["%s_avg" % col] - train[col]
        if dev is not None:
            features_dev["delta_%s" % col] = dev["%s_avg" % col] - dev[col]
    # scaled it
    scaler = preprocessing.StandardScaler()
    features_train[["delta_%s" % col for col in cols["temporal"]]] = \
        scaler.fit_transform(
            features_train[["delta_%s" % col for col in cols["temporal"]]])
    if dev is not None:
        features_dev[["delta_%s" % col for col in cols["temporal"]]] = \
            scaler.transform(
                features_dev[["delta_%s" % col for col in cols["temporal"]]])
    return features_train, features_dev


def add_temporal_rolling_mean(delta, train, features_train,
                              dev=None, features_dev=None,
                              columns=None):
    """ """
    if columns is None:
        columns = cols["temporal"]
    # compute rolling mean of step delta
    rol_df = train.groupby("block")[columns] \
        .rolling(delta, min_periods=0).mean() \
        .reset_index(0, drop=True) \
        .sort_index()
    rol_df.rename(
        columns=dict((col, "%s_mean_%i" % (col, delta))
                     for col in columns),
        inplace=True)
    features_train = features_train.merge(
        rol_df, left_index=True, right_index=True, copy=False)
    if dev is not None:
        rol_df = dev.groupby("block")[columns] \
            .rolling(delta, min_periods=0).mean() \
            .reset_index(0, drop=True) \
            .sort_index()
        rol_df.rename(
            columns=dict((col, "%s_mean_%i" % (col, delta))
                         for col in columns),
            inplace=True)
        features_dev = features_dev.merge(
            rol_df, left_index=True, right_index=True,
            suffixes=("", "_mean_%i" % delta))
    # scale it
    # scaler = preprocessing.RobustScaler()
    # features_train[
    #     ["%s_mean_%i" % (col, delta) for col in cols["temporal"]]
    # ] = scaler.fit_transform(
    #     features_train[["%s_mean_%i" % (col, delta)
    #                     for col in cols["temporal"]]])
    # if dev is not None:
    #     features_dev[
    #         ["%s_mean_%i" % (col, delta) for col in cols["temporal"]]
    #     ] = scaler.transform(
    #         features_dev[["%s_mean_%i" % (col, delta)
    #                       for col in cols["temporal"]]])
    return features_train, features_dev


def rolling_mean_col(df, delta, cols):
    """ """
    if isinstance(cols, basestring):
        cols = [cols]
    # compute rolling mean of step delta
    rol_df = df.groupby("block")[cols] \
        .rolling(delta, min_periods=0).mean() \
        .reset_index(0, drop=True) \
        .sort_index()
    rol_df.rename(
        columns=dict((col, "%s_mean_%i" % (col, delta)) for col in cols),
        inplace=True)
    return rol_df


def add_temporal_rolling_std(delta, train, features_train,
                             dev=None, features_dev=None):
    """ """
    # compute rolling mean of step delta
    rol_df = train.groupby("block")[cols["temporal"]] \
        .rolling(delta, min_periods=0).std() \
        .fillna(method="bfill") \
        .reset_index(0, drop=True) \
        .sort_index()
    rol_df.rename(
        columns=dict((col, "%s_std_%i" % (col, delta))
                     for col in cols["temporal"]),
        inplace=True)
    features_train = features_train.merge(
        rol_df, left_index=True, right_index=True, copy=False)
    if dev is not None:
        rol_df = dev.groupby("block")[cols["temporal"]] \
            .rolling(delta, min_periods=0).std() \
            .fillna(method="bfill") \
            .reset_index(0, drop=True) \
            .sort_index()
        rol_df.rename(
            columns=dict((col, "%s_std_%i" % (col, delta))
                         for col in cols["temporal"]),
            inplace=True)
        features_dev = features_dev.merge(
            rol_df, left_index=True, right_index=True,
            suffixes=("", "_std_%i" % delta))
    # scale it
    # scaler = preprocessing.RobustScaler()
    # features_train[
    #     ["%s_std_%i" % (col, delta) for col in cols["temporal"]]
    # ] = scaler.fit_transform(
    #     features_train[["%s_std_%i" % (col, delta)
    #                     for col in cols["temporal"]]])
    # if dev is not None:
    #     features_dev[
    #         ["%s_std_%i" % (col, delta) for col in cols["temporal"]]
    #     ] = scaler.transform(
    #         features_dev[["%s_std_%i" % (col, delta)
    #                       for col in cols["temporal"]]])
    return features_train, features_dev


def add_temporal_shift(config, features_train, features_dev=None):
    """ """
    for col, delays in config.items():
        for delay in delays:
            features_train["%s_shift_%i" % (col, delay)] = \
                features_train[col].shift(delay).fillna(method="bfill")
            if features_dev is not None:
                features_dev["%s_shift_%i" % (col, delay)] = \
                    features_dev[col].shift(delay).fillna(method="bfill")
    #
    return features_train, features_dev


def add_temporal_decomposition(freq, train, features_train, dev=None,
                               features_dev=None, scale=False):
    """ """
    temporal_train_dec = decompose_temporal(freq, train)
    # scale it
    if scale:
        scaler = preprocessing.StandardScaler(with_mean=True)
        temporal_train_dec[temporal_train_dec.columns] = \
            scaler.fit_transform(temporal_train_dec)
    # merge
    features_train = features_train.merge(
        temporal_train_dec, left_index=True, right_index=True)
    if dev is not None:
        temporal_dev_dec = decompose_temporal(freq, dev)
        # scale it
        if scale:
            temporal_dev_dec[temporal_dev_dec.columns] = \
                scaler.transform(temporal_dev_dec)
        # merge
        features_dev = features_dev.merge(
            temporal_dev_dec, left_index=True, right_index=True)
    return features_train, features_dev


def decompose_temporal(freq, train, log=True):
    """ """
    import statsmodels.api as sm
    hour = datetime.timedelta(hours=1)
    d = datetime.datetime(2014, 1, 1, 0, 0)
    decompose = []
    for k, g in train.groupby("block"):
        #
        g["daytime"] = g["daytime"].map(lambda x: d + int(x) * hour)
        tmp = g.set_index(pd.DatetimeIndex(g["daytime"]))
        acc = g[["daytime"]]
        for col in cols["temporal"]:
            dec = sm.tsa.seasonal_decompose(tmp[[col]], freq=freq)
            res = concat_decomposition(dec)
            res = res.fillna(0.)
            acc = acc.merge(res, left_index=False, right_index=True,
                            left_on="daytime")
        decompose.append(acc)
    return pd.concat(decompose, axis=0).drop("daytime", axis=1).sort_index()


def concat_decomposition(dec):
    """ """
    to_merge = [
        dec.seasonal.rename(
            index=str,
            columns={dec.seasonal.columns[0]: "%s_seasonal" % dec.seasonal.columns[0]}),
        dec.trend.rename(
            index=str,
            columns={dec.trend.columns[0]: "%s_trend" % dec.trend.columns[0]}),
        dec.resid.rename(
            index=str,
            columns={dec.resid.columns[0]: "%s_resid" % dec.resid.columns[0]}),
    ]
    res = pd.concat(to_merge, axis=1)
    # cast index to datetime
    res = res.set_index(pd.DatetimeIndex(res.index))
    return res


def to_log(df):
    """ """
    res = df.copy()
    for col in ["temperature", "pressure"]:
        res[col] = np.log(273.0 + df[col])
    return res


def hours_day(df):
    """ """
    return df.daytime.map(lambda x: (x - daytime_0) % 24)
    # df["day_of_year"] = df.hour_of_day.map(lambda x: x % 24)


def day_of_week(df):
    """ """
    return df.daytime.map(lambda x: ((x - daytime_0) // 24) % 7)


def normalize_df(df):
    """ """
    return pd.DataFrame(preprocessing.normalize(df), columns=df.columns,
                        index=df.index)


def drop_cols(df, cols):
    """ """
    return df[[col for col in df.columns if col not in cols]]


def make_features(train, dev=None, scale_temp=True,
                  rolling_mean=True, deltas_mean=[], roll_mean_conf={},
                  rolling_std=True, deltas_std=[],
                  shift_config={}, temp_dec_freq=0,
                  binary_static=False,
                  pollutant=False,
                  remove_temporal=False, log=False):
    """ """
    general_col = ["zone_id", "is_calmday", "daytime"]
    f_train = train[general_col]
    if pollutant:
        encoder = preprocessing.LabelEncoder()
        f_train["pollutant"] = encoder.fit_transform(train["pollutant"])
    # hour of day & day of week
    f_train["hour_of_day"] = hours_day(train)
    f_train["day_of_week"] = day_of_week(train)
    # day of week
    if dev is not None:
        f_dev = dev[general_col]
        if pollutant:
            f_dev["pollutant"] = encoder.fit_transform(dev["pollutant"])
        # hour of day & day of week
        f_dev["hour_of_day"] = hours_day(dev)
        f_dev["day_of_week"] = day_of_week(dev)
    else:
        f_dev = None
    # to log for temperature and pressure
    if log:
        train = to_log(train)
        if dev is not None:
            dev = to_log(dev)
    # scale temporal features with robust scaling
    if scale_temp:
        f_train, f_dev = scale_temporal(train, f_train, dev, f_dev)
    else:
        f_train[cols["temporal"]] = train[cols["temporal"]]
        if dev is not None:
            f_dev[cols["temporal"]] = dev[cols["temporal"]]
    # scale data with MaxAbsScaler to handle sparse static data
    f_train, f_dev = scale_static(train, f_train, dev, f_dev, log=log)
    # binary static features
    if binary_static:
        f_train, f_dev = binarize_static(train, f_train, dev, f_dev)
    # Rolling mean of step delta
    if rolling_mean:
        if roll_mean_conf:
            for delta, columns in roll_mean_conf.items():
                f_train, f_dev = add_temporal_rolling_mean(
                    delta, train, f_train, dev, f_dev, columns)
        else:
            for delta in deltas_mean:
                f_train, f_dev = add_temporal_rolling_mean(
                    delta, train, f_train, dev, f_dev)
    # Rolling Std of step deltas_std
    if rolling_std:
        for delta in deltas_std:
            f_train, f_dev = add_temporal_rolling_std(
                delta, train, f_train, dev, f_dev)
    # temporal shift
    if shift_config:
        f_train, f_dev = add_temporal_shift(shift_config, f_train, f_dev)
    # temporal decomposition
    if temp_dec_freq:
        f_train, f_dev = add_temporal_decomposition(
            temp_dec_freq, train, f_train, dev, f_dev)
    if remove_temporal:
        f_train = drop_cols(f_train, cols["temporal"])
        if dev is not None:
            f_dev = drop_cols(f_dev, cols["temporal"])
    # drop daytime col
    if "daytime" in f_train:
        f_train.drop("daytime", axis=1, inplace=True)
    if dev is not None:
        if "daytime" in f_dev:
            f_dev.drop("daytime", axis=1, inplace=True)
    #
    if dev is not None:
        return f_train, f_dev
    else:
        return f_train


def build_sequences(df, seq_length, pad=False, pad_value=0., norm=True):
    """ """
    seqs = []
    ids = np.empty(shape=[0])
    for _, g in df.groupby("block"):
        g["ID"] = g.index
        g = g.set_index("daytime").sort_index().drop("block", axis=1)
        array = g.drop("ID", axis=1).values
        ids.append(g.ID)
        np.concatenate((ids, g.ID.as_matrix()), axis=0)
        # L2 normalize
        if norm:
            array = preprocessing.normalize(array)
        # seqs = []
        for k in range(1, seq_length + 1):
            seqs.append(array[:k])
        for k in range(seq_length, array.shape[0]):
            seqs.append(array[k - seq_length:k])
    if pad:
        from keras.preprocessing.sequence import pad_sequences
        seqs = pad_sequences(seqs, maxlen=seq_length, dtype='float32',
                             padding='pre', truncating='pre', value=pad_value)
    return seqs, ids


def make_seqential_features(train, dev=None, seq_length=12, normalize=False,
                            temp_dec_freq=0, pollutant=False,
                            remove_temporal=False, log=False):
    """ """
    # make standard features
    f_train, f_dev = make_features(train, dev=dev, scale_temporal=True,
                                   temp_dec_freq=temp_dec_freq,
                                   pollutant=pollutant,
                                   remove_temporal=remove_temporal, log=log)
    # sequantialize
    # add block column
    f_train["block"] = train["block"]
    train_seqs, train_ids = build_sequences(f_train, seq_length=seq_length,
                                            pad=True, norm=normalize)
    if dev is not None:
        f_dev["block"] = dev["block"]
        dev_seqs, dev_ids = build_sequences(f_dev, seq_length=seq_length,
                                            pad=True, norm=normalize)
    # return
    if dev is not None:
        return train_seqs, train_ids, dev_seqs, dev_ids
    else:
        return train_seqs, train_ids


def make_hybrid_features(train, dev=None, seq_length=12, normalize=False,
                         temp_dec_freq=0, pollutant=False,
                         remove_temporal=False, log=False):
    """ """
    columns = ["daytime", "zone_id", "hour_of_day", "day_of_week",
               "is_calmday", "block"]
    if temp_dec_freq:
        remove_temporal = True
    # make standard features
    f_train, f_dev = make_features(train, dev=dev, scale_temporal=True,
                                   temp_dec_freq=temp_dec_freq,
                                   pollutant=pollutant,
                                   remove_temporal=remove_temporal, log=log)

    # add block column
    f_train["block"] = train["block"]
    # temporal features: sequential
    temp_cols = [col for col in cols["temporal"]]
    f_temp_train = f_train[columns + temp_cols].drop("zone_id", axis=1)
    train_temp_seqs, train_ids = build_sequences(
        f_temp_train, seq_length=seq_length, pad=True, norm=normalize)
    if dev is not None:
        f_dev["block"] = dev["block"]
        f_temp_dev = f_dev[columns + temp_cols].drop("zone_id", axis=1)
        dev_temp_seqs, dev_ids = build_sequences(
            f_temp_dev, seq_length=seq_length, pad=True, norm=normalize)
    # static features
    static_cols = ["%s_sc" % col for col in cols["static"]] + ["zone_id"]
    # add wind sin and cosin
    static_cols.extend("windbearingcos", "windbearingsin")
    train_static_ds = np.empty(shape=[0, len(static_cols)])
    gb = f_train.groupby("block")
    for k, group in gb:
        group = group.set_index("daytime").sort_index()
        train_static_ds = np.concatenate(
            (train_static_ds, group[static_cols].values), axis=0)
        if normalize:
            train_static_ds = preprocessing.normalize(train_static_ds)
    if dev is not None:
        dev_static_ds = np.empty(shape=[0, len(static_cols)])
        gb = f_dev.set_index("daytime").groupby("block")
        for k, group in gb:
            dev_static_ds = np.concatenate(
                (dev_static_ds, group[static_cols].values), axis=0)
        if normalize:
            dev_static_ds = preprocessing.normalize(dev_static_ds)
    # return
    if dev is not None:
        return [train_temp_seqs, train_static_ds, train_ids], [dev_temp_seqs, dev_static_ds, dev_ids]
    else:
        return [train_temp_seqs, train_static_ds, train_ids]


def get_seq_Y(X, Y, pollutant=None):
    """ """
    Y_seq = np.empty(shape=[0])
    X_u = X[["daytime", "block", "pollutant"]]
    Y_u = Y.merge(X_u, left_index=True, right_index=True, how="inner")
    # if no pollutant passed find it
    if pollutant is None:
        tmp = Y_u.pollutant.unique()
        if len(tmp) == 1:
            pollutant = tmp[0]
        else:
            raise ValueError(
                "Many pollutants in df, please set one in pollutant arg")
    gb = Y_u[Y_u["pollutant"] == pollutant].groupby("block")
    for k, group in gb:
        Y_seq = np.concatenate((Y_seq, group.TARGET.values))
    return Y_seq



if __name__ == '__main__':
    """ """
    pass



