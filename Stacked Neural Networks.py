import tensorflow as tf
import numpy as np
from numpy.random import seed
seed(1)
import keras
import sklearn
from keras.layers import Input, Embedding, Reshape, merge, Dropout, Dense, LSTM, core, Activation
from keras.layers import TimeDistributed, Flatten, concatenate, Bidirectional, Concatenate, Conv1D, MaxPooling1D, Conv2D
from keras.utils import np_utils
from keras.engine import Model
from keras.models import Sequential
from keras import layers, optimizers
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import argmax
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
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
from numpy import dstack
random.seed(42)
tf.random.set_seed(2)
#
sc = StandardScaler()

#
n_samples = 100000
kfold_splits = 5
n_epochs = 40

# data preparation
# load synthetic COVID-19 dataset generated from R
COVID_data = pd.read_csv('/Users/ryanjmulholland/documents/Data/covsynth.csv')
COVID_data = COVID_data.values

## define X and y
unique_IDs = COVID_data[:, 0]
X = COVID_data[:, 1:(COVID_data.shape[1] - 1)]
y = COVID_data[:, (COVID_data.shape[1] - 1)]

# define kfold splits - split the unique IDs to avoid peeking
from sklearn.model_selection import KFold
kf = KFold(n_splits= kfold_splits)
kf.get_n_splits(unique_IDs)
print(kf)

#
for index, (train_indices, val_indices) in enumerate(kf.split(unique_IDs)):
    print("Training on fold " + str(index+1) + "/" + str(kfold_splits) + "...")
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
    np.savetxt("/Users/ryanjmulholland/documents/Data/y_train_" + str(index) +  "_.csv", y_train, delimiter=",")
    np.savetxt("/Users/ryanjmulholland/documents/Data/X_val_" + str(index) +  "_.csv", X_val[:, 0], delimiter=",")
    np.savetxt("/Users/ryanjmulholland/documents/Data/y_val_" + str(index) +  "_.csv", y_val, delimiter=",")
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
#
## Feature connected layer setup
    dense_N = 36
    dropout_n = 0.8
    n_batch_size = 1024 # 64
#
    # COVID admission data
    admiss_data = Input(shape = (len(X_train[1]), ), dtype='float32')
#
    fcl_process = Dense(dense_N)(admiss_data)
    fcl_process = Dense(dense_N)(fcl_process)
    fcl_process = Dropout(dropout_n)(fcl_process)
    fcl_process = Dense(dense_N)(fcl_process)
    fcl_process = Dropout(dropout_n)(fcl_process)
#
    main_output = Dense(1, activation = 'sigmoid')(fcl_process)

#fit the model on the COVID dataset
def fit_model(X_train, y_train):
#define model
    model = Model(inputs=[admiss_data], outputs=main_output)
#
    print(model.summary())
#
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
    model.compile(optimizer = adam, loss='binary_crossentropy', metrics=['accuracy'])
#
    class_weight = {0: 1.,
                    1: cw, # 1: 20.
                    }
    histories = my_callbacks.Histories()
    #model fit
    model.fit([X_train], y_train, epochs=n_epochs, batch_size=n_batch_size, validation_data = ([[X_val], y_val]),  class_weight=class_weight, callbacks = [histories])
    model.save('base_nn.h5')
    return model


#fit and save nn base models
base_nns = 7
for i in range(base_nns):
    nn_model = fit_model(X_train, y_train)
    filename = '/Users/ryanjmulholland/documents/nnmodels/model_' + str(i + 1) + '.h5'
    nn_model.save(filename)


# load nns
def load_all_nns(n_nns):
    all_nns = list()
    for i in range(n_nns):
        # define the filename for stacked nn
        filename = '/Users/ryanjmulholland/documents/nnmodels/model_' + str(i + 1) + '.h5'
        # load nn
        model = load_model(filename)
        # add nn to assembled list
        all_nns.append(model)
        print('>loaded %s' % filename)
    return all_nns

base_nns=7
nns = load_all_nns(base_nns)


# generate ensemble model input dataset as outputs from stacked models
def stacked_dataset(base_nns, inpX):
    stackedX = None
    for model in nns:
        #predict
        ypred= model.predict(inpX, verbose=0)
        # stacking of predictions into [rows, base nns, probabilities]
        if stackedX is None:
            stackedX = ypred
        else:
            stackedX = dstack((stackedX, ypred))
    # flatten into [rows, base nns x probabilities]
    stackedX = stackedX.reshape((stackedX.shape[0], stackedX.shape[1] * stackedX.shape[2]))
    return stackedX


# fit model using outputs of base learners
def fit_stacked_model(base_nns, inpX, inpy):
    # generate dataset utilising the stacked model
    stackingX = stacked_dataset(base_nns, inpX)
    # fit stacked ensemble
    model = LogisticRegression()
    model.fit(stackingX, inpy)
    return model


# make a prediction with the stacked model
def stacked_prediction(base_nns, model, inpX):
    # create dataset using ensemble
    stackingX = stacked_dataset(base_nns, inpX)
    # make a prediction
    ypred = model.predict(stackingX)
    return ypred

# load all models
n_nns = 7
base_nns = load_all_nns(n_nns)
print('Loaded %d models' % len(base_nns))

# fit stacked model using the ensemble
model = fit_stacked_model(base_nns, X_val, y_val)

# evaluate model on test set
ypred = stacked_prediction(base_nns, model, X_val)
acc = accuracy_score(y_val, ypred)

def get_model():
    return load_model('base_nn.h5')

stk_nn =KerasClassifier(build_fn=get_model)

classif= [stk_nn]

kf = model_selection.StratifiedKFold(n_splits=5)

for i, ensem in enumerate(classif):
    cvscore = model_selection.cross_val_score(ensem, X_train, y_train,
    cv=kf, scoring='accuracy')
    print("Stacked Ensemble Model %0.0f" % i)
    print("Train (CV) Acc: %0.2f (+/- %0.2f)" % (cvscore.mean(),
    cvscore.std()))
    ensem.fit(X_train, y_train)
    print("Train Acc: %0.2f " % (metrics.accuracy_score(ensem.predict
    (X_train), y_train)))

    print("Test Acc: %0.2f " % (metrics.accuracy_score(ypred, y_val)))
    conf=(confusion_matrix(y_val, ypred))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf)

    display = disp.plot(values_format='')
    name = 'conf' + str(i + 1) + '.svg'
    plt.savefig(name)
    y_probab = ensem.predict_proba(X_val)[::, 1]

    fprate, tprate, _ = roc_curve(y_val, y_probab)
    auc = roc_auc_score(y_val, y_probab)
    print(auc)

