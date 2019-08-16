# Data
from pandas import read_csv
from sklearn.metrics import precision_recall_curve	
from inspect import signature
import numpy as np
import pandas as pd
import xgboost as xgb 
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import regularizers
from keras import optimizers
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NearMiss
from keras import backend as K
import timeit

def feature_select(data,target,n_feat):
	from sklearn.linear_model import LogisticRegression

	model = LogisticRegression(class_weight=class_weight).fit(data,target)

	from sklearn.feature_selection import SelectFromModel

	selection = SelectFromModel(model,max_features=n_feat,prefit=True)
	new_data = selection.transform(data)

	return new_data


def logi(X_train,X_test,y_train,y_test):
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(class_weight=class_weight).fit(X_train,y_train)

	y_pred = model.predict(X_test)

	return y_pred


def boost(X_train,X_test,y_train,y_test):
	ada = ADASYN(sampling_strategy='minority')
	X_train_, y_train_ = ada.fit_resample(X_train,y_train)
	print(X_train_.shape)

	from numpy import sort
	from sklearn.feature_selection import SelectFromModel

	param = {'max_depth':9, 'eta':0.7, 'silent':1, 'objective':'binary:logistic'}
	param['booster']='dart'
	param['nthread'] = 4
	param['silent'] = 1
	param['eval_metric'] = 'auc'

	model = xgb.XGBClassifier(params=param)
	model.fit(X_train_, y_train_)

	y_pred = model.predict(X_test)


	return y_pred

def deep_learn3(X_train,X_test,y_train,y_test):

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)


	nb_epoch = 150
	batch_size = 256
	input_dim = X_train.shape[1]
	learning_rate = 1e-3
	decay = learning_rate/nb_epoch

	input_layer = Input(shape=(input_dim, ))

	net = Dense(input_dim,activation="linear",activity_regularizer=regularizers.l1(learning_rate))(input_layer)
	net = Dense(15,activation="linear")(net)
	net = Dense(10,activation="linear")(net)
	net = Dense(5,activation="linear")(net)

	output_layer = Dense(1, activation='sigmoid')(net)

	adam = optimizers.Adam(lr=learning_rate,decay=decay)

	model = Model(inputs=input_layer, outputs=output_layer)
	model.compile(metrics=['accuracy'],
                    loss='binary_crossentropy',
                    optimizer=adam)


	cp = ModelCheckpoint(filepath="NeuralNetworkModel.h5",
	                               save_best_only=True,
	                               verbose=0)

	tb = TensorBoard(log_dir='./logs',
	                histogram_freq=0,
	                write_graph=True,
	                write_images=True)

	history = model.fit(X_train, y_train,
	                    epochs=nb_epoch,
	                    batch_size=batch_size,
	                    shuffle=True,
	                    validation_data=(X_test, y_test),
	                    verbose=1,
	                    class_weight=class_weight,
	                    callbacks=[cp, tb]).history



	# load weights
	model.load_weights("NeuralNetworkModel.h5")
	# Compile model (required to make predictions)
	model.compile(metrics=['accuracy'],
                    loss='binary_crossentropy',
                    optimizer=adam)

	y_pred = model.predict(X_test)

	return y_pred

def deep_learn2(X_train,X_test,y_train,y_test):

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	nb_epoch = 80
	batch_size = 64
	input_dim = X_train.shape[1]
	# learning_rate = 1
	# decay = learning_rate/nb_epoch

	input_layer = Input(shape=(input_dim, ))

	net = Dense(200,activation="relu",activity_regularizer=regularizers.l2(1e-7))(input_layer)
	net = Dense(400, activation="relu")(net)
	net = Dense(600, activation="relu")(net)
	net = Dense(800, activation="relu")(net)
	net = Dense(1000,activation="relu")(net)
	net = Dense(800,activation="relu")(net)
	net = Dense(600, activation="relu")(net)
	net = Dense(400, activation="relu")(net)
	net = Dense(200, activation="relu")(net)

	output_layer = Dense(1, activation='sigmoid')(net)

	adam = optimizers.Adam(lr=1e-5)

	model = Model(inputs=input_layer, outputs=output_layer)
	model.compile(metrics=['accuracy'],
                    loss='binary_crossentropy',
                    optimizer=adam)


	cp = ModelCheckpoint(filepath="NeuralNetworkModel.h5",
	                               save_best_only=True,
	                               verbose=0)

	tb = TensorBoard(log_dir='./logs',
	                histogram_freq=0,
	                write_graph=True,
	                write_images=True)

	history = model.fit(X_train, y_train,
	                    epochs=nb_epoch,
	                    batch_size=batch_size,
	                    shuffle=True,
	                    validation_data=(X_test, y_test),
	                    verbose=1,
	                    class_weight=class_weight,
	                    callbacks=[cp, tb]).history

	# load weights
	model.load_weights("NeuralNetworkModel.h5")
	# Compile model (required to make predictions)
	model.compile(metrics=['accuracy'],
                    loss='binary_crossentropy',
                    optimizer=adam)
	y_pred = model.predict(X_test)

	return y_pred

DATA = read_csv("norm_data__non_log.txt",sep='\t').T
DATA = DATA.apply(np.log).values # Retain the log due to the maximising values
label = read_csv("sample_list.csv",sep=';')

# Conversion of string to bool
mapping = {'Non-LCa':0,'LCa':1}
TARGET = label.Disease.map(mapping).values
class_weight = {1:2,0:1}

recall = []
precision = []
accuracy = []
auc = []
time = []

with open('final_results_dn2lr1e5_cv10_76.txt', 'w') as f:
	print("FULL FINAL RESULTS \n\n",file=f)
	print("--------&&&&&&&&&&&&&&--------",file=f)
	for i in [76]:
		print("N_COMPONENTS = {} \n\n".format(i),file=f)
		print("CV = 5 \n\n",file=f)
		# Booster
		new_data = feature_select(DATA,TARGET,i)
		skf = StratifiedKFold(n_splits=5)

		for train,test in skf.split(new_data,TARGET):
			X_train, X_test = new_data[train], new_data[test]
			y_train, y_test = TARGET[train], TARGET[test]
		

			start = timeit.default_timer()
			y_net2 = deep_learn2(X_train, X_test, y_train, y_test)
			stop = timeit.default_timer()

			time.append(stop - start)
			recall.append(recall_score(y_test,np.round(y_net2,0)))
			precision.append(precision_score(y_test,np.round(y_net2,0)))
			accuracy.append(accuracy_score(y_test,np.round(y_net2,0)))
			auc.append(roc_auc_score(y_test,y_net2))


		print("---------------------------------",file=f)
		print("RECALL {}\n".format(np.mean(recall)),file=f)
		print("PRECISION {}\n".format(np.mean(precision)),file=f)
		print("ACCURACY {}\n".format(np.mean(accuracy)),file=f)
		print("AUC {}\n".format(np.mean(auc)),file=f)
		print("TIME {}\n".format(np.mean(time)),file=f)
		print("---------------------------------",file=f)


