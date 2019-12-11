#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, Merge
from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error

# data
from dataset import (
    split_pollutant_dataset, preprocess_dataset, split_train_dev,
    get_Y, zone_station_train, zone_station_dev
)
from features import (
    make_features, make_seqential_features, get_seq_Y,
    make_hybrid_features
)


# tensorboard
log_dir = "./logs"
tensordboard_cb = TensorBoard(log_dir)


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


TIME_DELAY = 12

# Paths
X_train_path = "/Users/thomasopsomer/data/plume-data/X_train.csv"
X_test_path = "/Users/thomasopsomer/data/plume-data/X_test.csv"
Y_train_path = "/Users/thomasopsomer/data/plume-data/Y_train.csv"


# prepare data / features
df = pd.read_csv(X_train_path)
df = preprocess_dataset(df)
Y = pd.read_csv(Y_train_path)


# split for each pollutant
NO2_df, PM10_df, PM25_df = split_pollutant_dataset(df)


# split in train / dev for each pollutant
NO2_train, NO2_dev = split_train_dev(
    NO2_df, zone_station_train, zone_station_dev)
# get sequential feature set
NO2_seq_train, NO2_seq_dev = make_seqential_features(
    NO2_train, NO2_dev, seq_length=TIME_DELAY, normalize=True)
# get Y for sequential view
Y_seq_NO2_train = get_seq_Y(NO2_train, Y)
Y_seq_NO2_dev = get_seq_Y(NO2_dev, Y)


# split in train / dev for each pollutant
PM25_train, PM25_dev = split_train_dev(
    PM25_df, zone_station_train, zone_station_dev)
# get sequential feature set
PM25_seq_train, PM25_seq_dev = make_seqential_features(
    PM25_train, PM25_dev, seq_length=TIME_DELAY, normalize=True)
# get Y for sequential view
Y_seq_PM25_train = get_seq_Y(PM25_train, Y)
Y_seq_PM25_dev = get_seq_Y(PM25_dev, Y)


# Set up the model
INPUT_SHAPE = (12, 37)
LSTM_SIZE = 80

model = Sequential()
model.add(LSTM(LSTM_SIZE, dropout_W=0.2, dropout_U=0.2,
               input_shape=INPUT_SHAPE, consume_less="cpu"))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')


# Train the model
# model.fit(NO2_seq_train, Y_seq_NO2_train,
#           nb_epoch=100, batch_size=32, verbose=2)


model.fit(NO2_seq_train, Y_seq_NO2_train,
          nb_epoch=100, batch_size=32, verbose=2,
          validation_data=(NO2_seq_dev, Y_seq_NO2_dev),
          callbacks=[tensordboard_cb])



model.fit(PM25_seq_train, Y_seq_PM25_train,
          nb_epoch=20, batch_size=32, verbose=2,
          validation_data=(PM25_seq_dev, Y_seq_PM25_dev),
          callbacks=[tensordboard_cb])

model.save("model/pm25_lstm.h5")

# Evaluate
evaluate_mse(model, NO2_seq_train, NO2_seq_dev,
             Y_seq_NO2_train, Y_seq_NO2_dev)
evaluate_mse(model, PM25_seq_train, PM25_seq_dev,
             Y_seq_PM25_train, Y_seq_PM25_dev)



# Double LSTM
INPUT_SHAPE = (12, 37)
LSTM_SIZE_1 = 80
LSTM_SIZE_2 = 40

model = Sequential()
model.add(LSTM(LSTM_SIZE_1, dropout_W=0.2, dropout_U=0.2,
               input_shape=INPUT_SHAPE, consume_less="cpu",
               return_sequences=True))
model.add(LSTM(LSTM_SIZE_2, dropout_W=0.2, dropout_U=0.2,
               input_shape=INPUT_SHAPE, consume_less="cpu"))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')

# training
model.fit(PM25_seq_train, Y_seq_PM25_train,
          nb_epoch=20, batch_size=32, verbose=2,
          validation_data=(PM25_seq_dev, Y_seq_PM25_dev),
          callbacks=[tensordboard_cb])


# hybrid data

# split in train / dev for each pollutant
PM25_train, PM25_dev = split_train_dev(
    PM25_df, zone_station_train, zone_station_dev)
# get sequential feature set
PM25_seq_stat_train, PM25_seq_stat_dev = make_hybrid_features(
    PM25_train, PM25_dev, seq_length=TIME_DELAY, normalize=True)
# get Y for sequential view
Y_seq_PM25_train = get_seq_Y(PM25_train, Y)
Y_seq_PM25_dev = get_seq_Y(PM25_dev, Y)


# Combining Temporal And Static Features

TEMP_INPUT_SHAPE = (12, 18)
NUM_STATIC_FEATURES = 19
LSTM_SIZE = 36

# temporal features into RNN (here LSTM)
temporal_model = Sequential()
temporal_model.add(LSTM(LSTM_SIZE, dropout_W=0.3, dropout_U=0.3,
                   input_shape=TEMP_INPUT_SHAPE, consume_less="cpu"))
temporal_model.add(Dense(1, activation='relu'))

# static features into Dense or MLP
static_model = Sequential()
# static_model.add(Dense(output_dim=10, input_dim=NUM_STATIC_FEATURES, activation='relu'))
static_model.add(Activation('linear', input_shape=(NUM_STATIC_FEATURES,)))
#model.add(Dropout(0.4))
# static_model.add(Dense(output_dim=50, input_dim=100))

# merge both model
model = Sequential()
model.add(Merge([temporal_model, static_model], mode='concat'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

# compile model
model.compile(loss='mean_squared_error', optimizer='adam')


# training
model.fit(PM25_seq_stat_train, Y_seq_PM25_train,
          nb_epoch=20, batch_size=32, verbose=2,
          validation_data=(PM25_seq_stat_dev, Y_seq_PM25_dev),
          callbacks=[tensordboard_cb])



# compare xgboost

NO2_train, NO2_dev = split_train_dev(
    NO2_df, zone_station_train, zone_station_dev)

NO2_train_f, NO2_dev_f = make_features(NO2_train, NO2_dev,
                                       rolling_mean=True, deltas=[24, 36, 48, 96])
Y_NO2_train = get_Y(Y, NO2_train)
Y_NO2_dev = get_Y(Y, NO2_dev)


PM25_train, PM25_dev = split_train_dev(
    PM25_df, zone_station_train, zone_station_dev)

PM25_train_f, PM25_dev_f = make_features(PM25_train, PM25_dev, rolling_mean=True, deltas=[24, 36, 48, 96])
Y_PM25_train = get_Y(Y, PM25_train)
Y_PM25_dev = get_Y(Y, PM25_dev)


import xgboost as xgb
from sklearn.preprocessing import normalize

# min_child_weight

xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=200)

xgb_model.fit(PM25_train_f, Y_PM25_train)
xgb_model.fit(NO2_train_f, Y_NO2_train,
              eval_set=(NO2_dev_f, Y_NO2_dev),
              eval_metric="mse")

evaluate_mse(xgb_model, PM25_train_f, PM25_dev_f,
             Y_PM25_train, Y_PM25_dev)

evaluate_mse(xgb_model, NO2_train_f, NO2_dev_f,
             Y_NO2_train, Y_NO2_dev)










# Train
# model.fit([X_train_temp, X_train_static], Y_train)


