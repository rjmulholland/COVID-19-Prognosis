import numpy as np
from numpy.random import seed

seed(1)
import sklearn
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from numpy import argmax
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
import xgboost as xgb
import random
import pydot
import graphviz
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import impyute as impy
import my_callbacks
import os
import warnings

warnings.filterwarnings("ignore")

#
sc = StandardScaler()

#
n_samples = 100000
kfold_splits = 5
n_epochs = 40

# data preparation
# load COVID-19 dataset
COVID_data = pd.read_csv('~/documents/Data/covid_data.csv')
COVID_data = COVID_data.values

## define X and y
unique_IDs = COVID_data[:, 0]
X = COVID_data[:, 1:(COVID_data.shape[1] - 1)]
y = COVID_data[:, (COVID_data.shape[1] - 1)]

# define kfold splits - split the unique IDs to avoid peeking
from sklearn.model_selection import KFold

kf = KFold(n_splits=kfold_splits)
kf.get_n_splits(unique_IDs)
print(kf)

#
for index, (train_indices, val_indices) in enumerate(kf.split(unique_IDs)):
    print("Training on fold " + str(index + 1) + "/" + str(kfold_splits) + "...")
    print("TRAIN:", train_indices, "TEST:", val_indices)
#
    # ID list from indices:
    train_IDs = unique_IDs[train_indices]
    val_IDs = unique_IDs[val_indices]
#
    # generate sets from intersections
    intersection_train = np.isin(unique_IDs, train_IDs)
    intersection_val = np.isin(unique_IDs, val_IDs)
#
    # Generate batches from indices
    X_train, X_val = X[intersection_train], X[intersection_val]
    y_train, y_val = y[intersection_train], y[intersection_val]
    #
    # generate class weight
    cw = np.round(y_train.shape[0] / np.sum(y_train), 0)
    #
    # save out files with linkids for output analysis
    np.savetxt("~/documents/Data/y_train_" + str(index) + "_.csv", y_train, delimiter=",")
    np.savetxt("~/documents/Data/X_val_" + str(index) + "_.csv", X_val[:, 0], delimiter=",")
    np.savetxt("~/documents/Data/y_val_" + str(index) + "_.csv", y_val, delimiter=",")
    #
    # retain and then remove ID cols
    X_train_IDs, X_val_IDs = unique_IDs[intersection_train], unique_IDs[intersection_val]
    #   X_train, X_val = X_train[: , 1:X_train.shape[1]], X_val[:, 1:X_val.shape[1]]
    #   y_train, y_val = y_train[:, 0], y_val[:, 0]
    #
    #   from sklearn.preprocessing import StandardScaler
    scalers = {}
    scalers = StandardScaler()
    X_train = scalers.fit_transform(X_train)
    X_val = scalers.transform(X_val)

# Logistic regression
clf_lr = LogisticRegression()

# XGBoost Model
clf_XGB = XGBClassifier(n_estimators=150, max_depth=3)

# AdaBoost Model

clf_ada = AdaBoostClassifier(n_estimators=100)

# RF Model
clf_rf = RandomForestClassifier(n_estimators=500, max_features=0.25, criterion="entropy")

classif = [clf_XGB, clf_rf, clf_lr, clf_ada]

# define kfold
kfold = model_selection.StratifiedKFold(n_splits=5)

perform_table = pd.DataFrame(columns=['classif', 'fpr', 'tpr', 'auc'])

# Base model performance
for i, ensem in enumerate(classif):
    cvscore = model_selection.cross_val_score(ensem, X_train, y_train,
                                              cv=kfold, scoring='accuracy')
    print("Stacked Ensemble Model Base Learner %0.0f" % i)
    print("Train (CV) Acc: %0.2f (+/- %0.2f)" % (cvscore.mean(),
                                                 cvscore.std()))
    ensem.fit(X_train, y_train)
    print("Train Acc: %0.2f " % (metrics.accuracy_score(ensem.predict
                                                        (X_train), y_train)))
    ypred = ensem.predict(X_val)
    print("Test Acc: %0.2f " % (metrics.accuracy_score(ypred, y_val)))
    conf = (confusion_matrix(y_val, ypred))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf)

    display = disp.plot(values_format='')
    name = 'conf' + str(i + 1) + '.svg'
    plt.savefig(name)
    y_probab = ensem.predict_proba(X_val)[::, 1]

    fprate, tprate, _ = roc_curve(y_val, y_probab)
    auc = roc_auc_score(y_val, y_probab)
    print(auc)
    perform_table = perform_table.append({'classif': ensem.__class__.__name__,
                                          'fpr': fprate,
                                          'tpr': tprate,
                                          'auc': auc}, ignore_index=True)

# base models
estimators = [
    ('lr', clf_lr),
    ('rf', clf_rf),
    ('ada', clf_ada),
    ('xgb', clf_XGB),
]

# Generate and fit stacking ensemble
stclf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), passthrough=True)
scores = model_selection.cross_val_score(stclf, X_train, y_train, cv=kfold, scoring='accuracy')
stclf.fit(X_train, y_train)

# Set name of the classifiers as index labels
perform_table.set_index('classif', inplace=True)

# Stacking ensemble performance
print("Train CV Acc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("Train Acc: %0.2f " % (metrics.accuracy_score(stclf.predict(X_train), y_train)))
print("Test Acc: %0.2f " % (metrics.accuracy_score(stclf.predict(X_val), y_val)))
y_pred = stclf.predict(X_val)
conf = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf)
display = disp.plot(values_format='')
plt.savefig('stackedconf.svg')
y_proba = stclf.predict_proba(X_val)[::, 1]
fprate, tprate, _ = roc_curve(y_val, y_proba)
auc = roc_auc_score(y_val, y_proba)
print(auc)
perform_table = perform_table.append({'classif': stclf.__class__.__name__,
                                     'fpr': fprate,
                                     'tpr': tprate,
                                     'auc': auc}, ignore_index=True)

fig = plt.figure(figsize=(8, 6))

for i in perform_table.index:
    plt.plot(perform_table.loc[i]['fpr'],
             perform_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, perform_table.loc[i]['auc']))

# Rename index labels
perform_table.rename(index={'StackingClassifier': 'Stacked Ensemble', 'LogisticRegression': 'Logistic Regression',
                            'XGBClassifier': 'XGBoost', 'RandomForestClassifier': 'Random Forests',
                            'AdaBoostClassifier': 'AdaBoost'}, inplace=True)

# Plot AUC Curves
plt.plot([0, 1], [0, 1], color='purple', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("1-Specificity", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("Sensitivity", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')

plt.savefig('auc.svg')
plt.show()

# Calibration

calstclf = CalibratedClassifierCV(stclf, cv='prefit')
calstclf.fit(X_train, y_train)

# Calibration curve

plt.clf()
proba = calstclf.predict_proba(X_val)
yproba = proba[:, 1]

ax = plt.gca()

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])

ax.plot([0, 1], [0, 1], "k:")

calstclf_score = brier_score_loss(y_val, yproba, pos_label=1)
fraction_of_positives, mean_predicted_value = calibration_curve(y_val, yproba, n_bins=20)
ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Stacked Ensemble Calibration")

plt.title('Stacked Ensemble Calibration Curve', size=16)
plt.savefig('calib.svg')
plt.show()

