# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:23:33 2020

@author: stife
"""

#%% Загрузка

import os

PATH = os.path.join("datasets", "wine")
NAME = "winequality-red.csv"

import pandas as pd

def load_data(path = PATH, name = NAME): # вернуть DataFrame из csv
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)

wine = load_data()

#%% обзор данных в консоли

pd.options.display.max_columns = wine.shape[1]
print(wine.head())
print(wine.info())

print(wine.describe())

#%% обзор данных в гистограммах
import matplotlib.pyplot as plt

wine.hist(bins = 8, figsize = (20, 20)) 
plt.show()

#%% проследим зависимости

corr_matrix = wine.corr()

print(corr_matrix["density"].sort_values(ascending = False))

#%% проследим зависимости

from pandas.plotting import scatter_matrix

attributes = ["density", "fixed acidity", "citric acid", "residual sugar", "pH", "alcohol"] # наблюдаем хорошую линейную зависимость у fixed acidity и alcohol 
scatter_matrix(wine[attributes], figsize = (30, 30))

#%% попробуем понять как зависят "подозрительные аттрибуты"

print(corr_matrix["citric acid"]["fixed acidity"])

print(corr_matrix["total sulfur dioxide"]["free sulfur dioxide"])

#%% пробуем визуализировать данные - наблюдаем хорошую тенденденцию к разделению на страты

wine.plot(kind = "scatter", x = "fixed acidity", y = "alcohol", alpha = 0.6,
          s = wine["quality"] * 5, label = "quality", figsize = (10, 10),
          c = "density", cmap = plt.get_cmap ("jet"), colorbar = True)
plt.legend()

#%% уберем ненужные признаки

del wine["total sulfur dioxide"]
del wine["volatile acidity"]
del wine["free sulfur dioxide"]

#%% отделим тестовую выборку от тренировочной с помощью стратифицированной выборки по quality

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
sss.get_n_splits(wine)

for train_i, test_i in sss.split(wine, wine["quality"]):
    train_set = wine.loc[train_i]
    test_set = wine.loc[test_i]

train_set.info()
test_set.info()

print(train_set["quality"].value_counts())
print(test_set["quality"].value_counts())

#%% выполним масштабирование по минимаксу

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(train_set)
train_set = scaler.transform(train_set)

print(train_set)

