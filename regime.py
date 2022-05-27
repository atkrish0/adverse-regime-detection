# check unit roots in time series

import datetime
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

# remove columns with missing observations
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

# check stationarity
threshold = 0.01  # significance level
for column in bigmacro.drop(['Date', 'Regime'], axis=1):
    result = adfuller(bigmacro[column])
    if result[1] > threshold:
        bigmacro[column] = bigmacro[column].diff()
bigmacro = bigmacro.dropna(axis=0)

# standardize
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

# time Series Split
df_idx = df[df.Date == '12/1/96'].index[0]

df_targets = df['Label'].values
df_features = df.drop(['Regime', 'Date', 'Label'], axis=1)

df_training_features = df.iloc[:df_idx, :].drop(['Regime', 'Date', 'Label'], axis=1)
df_validation_features = df.iloc[df_idx:, :].drop(['Regime', 'Date', 'Label'], axis=1)

df_training_targets = df['Label'].values
df_training_targets = df_training_targets[:df_idx]

df_validation_targets = df['Label'].values
df_validation_targets = df_validation_targets[df_idx:]

print(len(df_training_features), len(df_training_targets), len(df_targets))
print(len(df_validation_features), len(df_validation_targets), len(df_features))

scoring = "roc_auc"
kfold = model_selection.TimeSeriesSplit(n_splits=3)
seed = 8

# create regularization hyperparameter space
C = np.reciprocal([0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005,
                   0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000])

# create hyperparameter options
hyperparameters = dict(C=C)

model = LogisticRegression(max_iter=10000, penalty='l1')
LR_penalty = model_selection.GridSearchCV(estimator=model, param_grid=hyperparameters,
                                          cv=kfold, scoring=scoring).fit(X=df_features,
                                                                         y=df_targets).best_estimator_


X = df_features
y = df_targets
lr_l1 = LogisticRegression(C=0.1, max_iter=10000, penalty="l1").fit(X, y)
model = SelectFromModel(lr_l1, prefit=True)
feature_idx = model.get_support()
feature_name = X.columns[feature_idx]
X_new = model.transform(X)

df_2 = df[feature_name]
df_2.insert(loc=0, column="Date", value=df['Date'].values)
df_2.insert(loc=1, column="Regime", value=df['Regime'].values)
df_2.insert(loc=2, column="Label", value=df['Label'].values)

corr = df_2.drop(['Date', 'Regime', 'Label'], axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)


# training Algorithms on Training Dataset
df = df_2

# time Series Split
df_idx = df[df.Date == '12/1/96'].index[0]

df_targets = df['Label'].values
df_features = df.drop(['Regime', 'Date', 'Label'], axis=1)

df_training_features = df.iloc[:df_idx, :].drop(['Regime', 'Date', 'Label'], axis=1)
df_validation_features = df.iloc[df_idx:, :].drop(['Regime', 'Date', 'Label'], axis=1)

df_training_targets = df['Label'].values
df_training_targets = df_training_targets[:df_idx]

df_validation_targets = df['Label'].values
df_validation_targets = df_validation_targets[df_idx:]

seed = 8
scoring = 'roc_auc'
kfold = model_selection.TimeSeriesSplit(n_splits=3)
models = []

models.append(('LR', LogisticRegression(C=1e09)))
models.append(('LR_L1', LogisticRegression(penalty='l1')))
models.append(('LR_L2', LogisticRegression(penalty='l2')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('ABC', AdaBoostClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('XGB', xgb.XGBClassifier()))

results = []
names = []
lb = preprocessing.LabelBinarizer()

for name, model in models:
    cv_results = model_selection.cross_val_score(estimator=model, X=df_training_features,
                                                 y=lb.fit_transform(df_training_targets), cv=kfold, scoring=scoring)

    model.fit(df_training_features, df_training_targets)  # train the model
    fpr, tpr, thresholds = metrics.roc_curve(
        df_training_targets, model.predict_proba(df_training_features)[:, 1])
    auc = metrics.roc_auc_score(
        df_training_targets, model.predict(df_training_features))
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name, auc))
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

fig = plt.figure()
fig.suptitle('Algorithm Comparison based on Cross Validation Scores')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# evaluate Performances of the Algorithms on Validation Dataset
model = LogisticRegression(C=1e09)  # high penalty
LR = model.fit(df_training_features, df_training_targets)
training_predictions = LR.predict(df_training_features)
prob_predictions = LR.predict_proba(df_training_features)
prob_predictions = np.append(prob_predictions, LR.predict_proba(df_validation_features), axis=0)

# define periods of recession
rec_spans = []

# rec_spans.append([datetime.datetime(1957,8,1), datetime.datetime(1958,4,1)])
rec_spans.append([datetime.datetime(1960, 4, 1),datetime.datetime(1961, 2, 1)])
rec_spans.append([datetime.datetime(1969, 12, 1),datetime.datetime(1970, 11, 1)])
rec_spans.append([datetime.datetime(1973, 11, 1),datetime.datetime(1975, 3, 1)])
rec_spans.append([datetime.datetime(1980, 1, 1),datetime.datetime(1980, 6, 1)])
rec_spans.append([datetime.datetime(1981, 7, 1),datetime.datetime(1982, 10, 1)])
rec_spans.append([datetime.datetime(1990, 7, 1),datetime.datetime(1991, 2, 1)])
rec_spans.append([datetime.datetime(2001, 3, 1),datetime.datetime(2001, 10, 1)])
rec_spans.append([datetime.datetime(2007, 12,1), datetime.datetime(2009,5,1)])

sample_range = pd.date_range(start='9/1/1960', end='9/1/2018', freq='MS')

plt.figure(figsize=(20, 5))
plt.plot(sample_range.to_series().values, prob_predictions[:, 0])
for i in range(len(rec_spans)):
    plt.axvspan(rec_spans[i][0], rec_spans[i][len(rec_spans[i]) - 1], alpha=0.25, color='grey')
plt.axhline(y=0.5, color='r', ls='dashed', alpha=0.5)
plt.title('Recession Prediction Probabalities with Logistic Regression')
mp.savefig('plot1.png',  bbox_inches='tight')
plt.show()

# create regularization penalty space
penalty = ['l1', 'l2']

# create regularization hyperparameter space
C = np.reciprocal([0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005,
                   0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000])

# create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

model = LogisticRegression(max_iter=10000)
LR_penalty = model_selection.GridSearchCV(estimator=model, param_grid=hyperparameters,
                                          cv=kfold, scoring=scoring).fit(df_training_features,
                                                                         df_training_targets).best_estimator_
training_predictions = LR_penalty.predict(df_training_features)

prob_predictions = LR_penalty.predict_proba(df_training_features)
prob_predictions = np.append(prob_predictions, LR_penalty.predict_proba(df_validation_features), axis=0)

sample_range = pd.date_range(start='9/1/1960', end='9/1/2018', freq='MS')

plt.figure(figsize=(20, 5))
plt.plot(sample_range.to_series().values, prob_predictions[:, 0])
for i in range(len(rec_spans)):
    plt.axvspan(rec_spans[i][0], rec_spans[i][len(rec_spans[i]) - 1], alpha=0.25, color='grey')
plt.axhline(y=0.5, color='r', ls='dashed', alpha=0.5)
plt.title('Recession Prediction Probabalities with Regularized Logistic Regression')
mp.savefig('plot2.png',  bbox_inches='tight')
plt.show()

# XGBoosting
xgboost = model_selection.GridSearchCV(estimator=xgb.XGBClassifier(),
                                       param_grid={'booster': ['gbtree']},
                                       scoring=scoring, cv=kfold).fit(df_training_features,
                                                                      lb.fit_transform(df_training_targets)).best_estimator_
xgboost.fit(df_training_features, df_training_targets)

prob_predictions = xgboost.predict_proba(df_training_features)
prob_predictions = np.append(prob_predictions, xgboost.predict_proba(df_validation_features), axis=0)

sample_range = pd.date_range(start='9/1/1960', end='9/1/2018', freq='MS')

plt.figure(figsize=(20, 5))
plt.plot(sample_range.to_series().values, prob_predictions[:, 0])
for i in range(len(rec_spans)):
    plt.axvspan(rec_spans[i][0], rec_spans[i]
                [len(rec_spans[i]) - 1], alpha=0.25, color='grey')
plt.axhline(y=0.5, color='r', ls='dashed', alpha=0.5)
plt.title('Recession Prediction Probabalities with XGBoost')
mp.savefig('plot3.png',  bbox_inches='tight')
plt.show()

# find feature importances
headers = df.drop(['Regime', 'Label', 'Date'], axis=1).columns.values.tolist()
xgboost_importances = pd.DataFrame(
    xgboost.feature_importances_, index=headers, columns=['Relative Importance'])
_ = xgboost_importances.sort_values(
    by=['Relative Importance'], ascending=False, inplace=True)
xgboost_importances = xgboost_importances[xgboost_importances['Relative Importance'] > 0].iloc[:20]

# display importances in bar-chart and pie-chart
fig = plt.figure(figsize=(6, 6))
plt.xticks(rotation='90')
plt.barh(y=np.arange(len(xgboost_importances)),
         width=xgboost_importances['Relative Importance'], align='center', tick_label=xgboost_importances.index)
plt.gca().invert_yaxis()
mp.savefig('feature_importance.png',  bbox_inches='tight')
plt.show()

# ROC AUC - Validation Targets
fpr, tpr, thresholds = metrics.roc_curve(df_validation_targets, LR.predict_proba(df_validation_features)[:, 1])
auc = metrics.roc_auc_score(df_validation_targets, LR.predict(df_validation_features))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LR', auc))

fpr, tpr, thresholds = metrics.roc_curve(df_validation_targets, LR_penalty.predict_proba(df_validation_features)[:, 1])
auc = metrics.roc_auc_score(df_validation_targets, LR_penalty.predict(df_validation_features))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LR_penalty', auc))

fpr, tpr, thresholds = metrics.roc_curve(df_validation_targets, xgboost.predict_proba(df_validation_features)[:, 1])
auc = metrics.roc_auc_score(df_validation_targets, xgboost.predict(df_validation_features))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('XGBoost', auc))

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic (Validation Data)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.savefig('ROC1.png',  bbox_inches='tight')
plt.show()

# ROC AUC - Actual Targets
fpr, tpr, thresholds = metrics.roc_curve(df_targets, LR.predict_proba(df_features)[:, 1])
auc = metrics.roc_auc_score(df_targets, LR.predict(df_features))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LR', auc))


fpr, tpr, thresholds = metrics.roc_curve(df_targets, LR_penalty.predict_proba(df_features)[:, 1])
auc = metrics.roc_auc_score(df_targets, LR_penalty.predict(df_features))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LR_penalty', auc))

fpr, tpr, thresholds = metrics.roc_curve(df_targets, xgboost.predict_proba(df_features)[:, 1])
auc = metrics.roc_auc_score(df_targets, xgboost.predict(df_features))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('XGBoost', auc))

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic (Whole period)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.savefig('ROC2.png',  bbox_inches='tight')
plt.show()
