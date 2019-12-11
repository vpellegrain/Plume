#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd
import numpy as np
from math import floor


cols = {
    "temporal": [
        u'precipintensity',
        u'precipprobability',
        u'temperature',
        u'cloudcover',
        u'pressure',
        u'windbearingcos',
        u'windbearingsin',
        u'windspeed',
    ],
    "static": [
        u'hlres_50',
        u'hlres_300',
        u'hlres_500',
        u'hlres_100',
        u'hlres_1000',
        #
        u'hldres_50',
        u'hldres_100',
        u'hldres_500',
        u'hldres_1000',
        #
        u'route_100',
        u'route_300',
        u'route_500',
        u'route_1000',
        #
        u'green_5000',
        u'natural_5000',
        u'port_5000',
        u'industry_1000',
        u'roadinvdist',
    ],
}


# station by zone for train / dev
zone_station_dev = [
    (0.0, 16.0),
    (1.0, 18.0),
    (3.0, 25.0),
    (4.0, 23.0),
    (5.0, 5.0),
]

zone_station_train = [
    (0.0, 17.0),
    (0.0, 20.0),
    (1.0, 1.0),
    (1.0, 22.0),
    (2.0, 26.0),
    (2.0, 28.0),
    (3.0, 6.0),
    (3.0, 9.0),
    (4.0, 4.0),
    (4.0, 10.0),
    (5.0, 8.0),
    (5.0, 11.0)
]

daytime_0 = 72.0

# split per pollutant
POLLUTANT = ['NO2', 'PM10', 'PM2_5']
ZONES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def get_count_by_station_by_zone(df):
    """ """
    res = df.groupby(["zone_id", "station_id"])[["ID"]] \
        .count().to_dict()
    return res


def get_zone_station(df):
    """ """
    # get dico of zone: [station_id,]
    zone_station = df.groupby(["zone_id", "station_id"]).groups.keys()
    Z = defaultdict(list)
    for zone, station in zone_station:
        Z[zone].append(station)
    return Z


def add_block(df):
    """ """
    df["block"] = df.pollutant.map(str) + \
        df.zone_id.map(lambda x: "-%s" % int(x)) + \
        df.station_id.map(lambda x: "-%s" % int(x))


def split_pollutant_dataset(df, pm=False):
    """ """
    # add block
    if "block" not in df:
        add_block(df)
    # split according to pollutant
    NO2_df = df[df.pollutant == "NO2"]
    if pm:
        PM_df = df[(df.pollutant == "PM10") | (df.pollutant == "PM2_5")]
        return NO2_df, PM_df
    else:
        PM10_df = df[df.pollutant == "PM10"]
        PM25_df = df[df.pollutant == "PM2_5"]
        return NO2_df, PM10_df, PM25_df


def split_train_dev(df, zone_station_train=None,
                    zone_station_dev=None,
                    seed=42):
    """
    split the train data set in to a train and dev dataset
    make sur that the dev dataset have different station
    than the train dataset.
    """
    # np.random.seed(seed)
    # get the name of the pollutant
    poll = df.pollutant.unique()[0]
    #
    if zone_station_train is None or zone_station_dev is None:
        # Z = get_zone_station(df)
        # Sample one station for each zone to put in dev set
        # other are in the train set
        # zone_station_train = []
        # zone_station_dev = []
        # for k, v in Z.items():
        #     n_station = len(v)
        #     i = np.random.randint(n_station)
        #     zone_station_dev.append((k, v.pop(i)))
        #     zone_station_train.extend([(k, s) for s in v])
        zone_station_train = []
        zone_station_dev = []
        d = get_poll_zone_station(df)
        for zone, stations in d[poll].items():
            n_station = len(stations)
            i = np.random.randint(n_station)
            print i
            zone_station_dev.append((zone, stations.pop(i)))
            zone_station_train.extend([(zone, s) for s in stations])
    # filter df on block column created with split_pollutant_dataset
    train_blocks = ["%s-%i-%i" % (poll, z, s) for z, s in zone_station_train]
    dev_blocks = ["%s-%i-%i" % (poll, z, s) for z, s in zone_station_dev]
    df_train = df[df.block.apply(lambda x: x in train_blocks)]
    df_dev = df[df.block.apply(lambda x: x in dev_blocks)]
    return df_train, df_dev


def get_poll_zone_station(df):
    """ """
    poll_zone_station = df.groupby(["pollutant", "zone_id", "station_id"]) \
        .groups.keys()
    d = defaultdict(lambda : defaultdict(list))
    for p, z, s in poll_zone_station:
        d[p][z].append(s)
    return d


def add_hours_day(df):
    """ """
    df["hour_of_day"] = df.daytime.map(lambda x: (x - daytime_0) % 24)
    # df["day_of_year"] = df.hour_of_day.map(lambda x: x % 24)


def add_day_of_week(df):
    """ """
    df["day_of_week"] = df.daytime.map(lambda x: ((x - daytime_0) // 24) % 7)


def add_avg_temporal_per_zone(df):
    """ """
    avg_temp = df.groupby(["zone_id", "daytime"]).mean()[cols["temporal"]]
    res = df.merge(avg_temp, left_on=["zone_id", "daytime"], right_index=True,
                   suffixes=("", "_avg"), copy=False)
    return res


def preprocess_dataset(df):
    """ """
    # add hour of day (0-23)
    add_hours_day(df)
    add_day_of_week(df)
    df.is_calmday = df.is_calmday.map(lambda x: 1 if x else -1)
    # add avg temporal value per zone
    # df = add_avg_temporal_per_zone(df)
    return df


def get_Y(Y_df, X_df):
    """ """
    # return pd.merge(Y_df, X_df[["ID"]], how='inner').set_index("ID")
    return Y_df.loc[Y_df.index.isin(X_df.index)]


if __name__ == '__main__':
    """ """
    # data path
    X_train_path = "C:/Users/Victor/Documents/MVA/Data Challenge/Plume/X_train.csv"
    X_test_path = "C:/Users/Victor/Documents/MVA/Data Challenge/Plume/X_test.csv"
    Y_train_path = "C:/Users/Victor/Documents/MVA/Data Challenge/Plume/Y_train.csv"

    # load all dataset
    df = pd.read_csv(X_train_path)
    df = preprocess_dataset(df)

    # split for each pollutant
    NO2_df, PM10_df, PM25_df = split_pollutant_dataset(df)

    # split in train / dev for each pollutant
    NO2_train = split_train_dev(NO2_df, zone_station_train, zone_station_dev)
    NO2_dev = split_train_dev(NO2_df, zone_station_train, zone_station_dev)

    PM10_train = split_train_dev(PM10_df, zone_station_train, zone_station_dev)
    PM10_dev = split_train_dev(PM10_df, zone_station_train, zone_station_dev)

    PM25_train = split_train_dev(PM25_df, zone_station_train, zone_station_dev)
    PM25_dev = split_train_dev(PM25_df, zone_station_train, zone_station_dev)




