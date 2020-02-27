#load libraries
# to check unit root in time series
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from matplotlib import pyplot as mp
import seaborn as sns
import os

bigmacro = pd.read_csv("Macroeconomic_Variables.csv")
bigmacro = bigmacro.rename(columns={'sasdate': 'Date'})
Recession_periods = pd.read_csv('Recession_Periods.csv')
bigmacro.insert(loc=1, column="Regime",
                value=Recession_periods['Regime'].values)

#remove columns with missing observations
missing_colnames = []
for i in bigmacro.drop(['Date', 'Regime'], axis=1):
    observations = len(bigmacro)-bigmacro[i].count()
    if (observations > 10):
        print(i+':'+str(observations))
        missing_colnames.append(i)

bigmacro = bigmacro.drop(labels=missing_colnames, axis=1)

#rows with missing values
bigmacro = bigmacro.dropna(axis=0)

# Add lags
for col in bigmacro.drop(['Date', 'Regime'], axis=1):
    for n in [3, 6, 9, 12, 18]:
        bigmacro['{} {}M lag'.format(col, n)] = bigmacro[col].shift(
            n).ffill().values

# 1 month ahead prediction
bigmacro["Regime"] = bigmacro["Regime"].shift(-1)

bigmacro = bigmacro.dropna(axis=0)

#check stationarity
threshold = 0.01  # significance level
for column in bigmacro.drop(['Date', 'Regime'], axis=1):
    result = adfuller(bigmacro[column])
    if result[1] > threshold:
        bigmacro[column] = bigmacro[column].diff()
bigmacro = bigmacro.dropna(axis=0)

# Standardize
features = bigmacro.drop(['Date', 'Regime'], axis=1)
col_names = features.columns

scaler = StandardScaler()
scaler.fit(features)
standardized_features = scaler.transform(features)
standardized_features.shape
df = pd.DataFrame(data=standardized_features, columns=col_names)
df.insert(loc=0, column="Date", value=bigmacro['Date'].values)
df.insert(loc=1, column='Regime', value=bigmacro['Regime'].values)

Label = df["Regime"].apply(lambda regime: 1. if regime == 'Normal' else 0.)
df.insert(loc=2, column="Label", value=Label.values)
