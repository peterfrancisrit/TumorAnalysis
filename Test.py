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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.metrics import recall_score, classification_report, auc, roc_auc_score, roc_curve
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

def feature_select(data,data_test,target,n_feat):

	from sklearn.feature_selection import SelectFdr, chi2, SelectFromModel,f_classif,SelectKBest,mutual_info_classif
	from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest
	# select = LogisticRegression(class_weight='balanced')
	selection = SelectKBest(chi2,k=n_feat).fit(data,target)
	new_data = selection.transform(data)
	new_data_test = selection.transform(data_test)

	return new_data, new_data_test


def logi(X_train,X_test,y_train,y_test):
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(class_weight=class_weight).fit(X_train,y_train)

	y_pred = model.predict(X_test)

	return y_pred
def svm(X_train,X_test,y_train,y_test):
	from sklearn.svm import SVC
	model = SVC(class_weight='balanced',C=100,gamma=0.001).fit(X_train,y_train)

	y_pred = model.predict(X_test)

	return y_pred


def boost(X_train,X_test,y_train,y_test):
	ada = ADASYN(sampling_strategy='minority')
	X_train_, y_train_ = ada.fit_resample(X_train,y_train)
	print(X_train_.shape)

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

	ada = ADASYN()
	X_train, y_train = ada.fit_resample(X_train,y_train)

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



with open('final_results_all.txt', 'w') as f:
	print("FULL FINAL RESULTS \n\n",file=f)
	print("--------&&&&&&&&&&&&&&--------",file=f)

	X_t, X_te, y_train, y_test = train_test_split(DATA,TARGET,shuffle=True,random_state=42)

	for i in ['All']:

		print("N_COMPONENTS = {} \n\n".format(i),file=f)
		# Booster
		X_train, X_test = X_t, X_te
		print("SHAPE X_TRAIN AFTER FEATURE SELECT {}".format(X_train.shape))

		# PRED
		start = timeit.default_timer()
		y_svm = svm(X_train, X_test, y_train, y_test)
		stop = timeit.default_timer()
		time_svm = stop - start

		start = timeit.default_timer()
		y_log = logi(X_train, X_test, y_train, y_test)
		stop = timeit.default_timer()
		time_log = stop - start

		start = timeit.default_timer()
		y_boost = boost(X_train, X_test, y_train, y_test)
		stop = timeit.default_timer()
		time_boost = stop - start

		start = timeit.default_timer()
		y_net2 = deep_learn2(X_train, X_test, y_train, y_test)
		stop = timeit.default_timer()
		time_dn2 = stop - start

		start = timeit.default_timer()
		y_net3 = deep_learn3(X_train, X_test, y_train, y_test)
		stop = timeit.default_timer()
		time_dn3 = stop - start

		# FPR TPR
		fpr_svm, tpr_svm, _ = roc_curve(y_test,y_svm)
		fpr_log, tpr_log, _ = roc_curve(y_test, y_log)
		fpr_boost, tpr_boost, _ = roc_curve(y_test, y_boost)
		fpr_net2, tpr_net2, _ = roc_curve(y_test, y_net2)
		fpr_net3, tpr_net3, _ = roc_curve(y_test, y_net3)

		# PRECISION RECALL
		precision_svm, recall_svm,_ = precision_recall_curve(y_test,y_svm)
		precision_log, recall_log, _ = precision_recall_curve(y_test, y_log)
		precision_boost, recall_boost, _ = precision_recall_curve(y_test, y_boost)
		precision_net2, recall_net2, _ = precision_recall_curve(y_test, y_net2)
		precision_net3, recall_net3, _ = precision_recall_curve(y_test, y_net3)




		# PLOT
		plt.style.use(['seaborn-colorblind'])
		plt.figure(1)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr_svm, tpr_svm, label='SVM',alpha=0.6,color='m')
		plt.plot(fpr_log, tpr_log, label='LR',alpha=0.6,color='r')
		plt.plot(fpr_boost, tpr_boost, label='Boost',alpha=0.6,color='b')
		plt.plot(fpr_net2, tpr_net2, label='Deep Net 2',alpha=0.6,color='y')
		plt.plot(fpr_net3, tpr_net3, label='Deep Net 3',alpha=0.6,color='g')
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.title('ROC curve Model Select {}, {}# Components'.format('LR_MODEL',i))
		plt.legend(loc='best')
		plt.savefig('ROCcurve_{}_{}.png'.format('LR_MODEL',i),dpi=300)
		plt.show()


		step_kwargs = ({'step': 'post'}
		               if 'step' in signature(plt.fill_between).parameters
		               else {})
		plt.step(recall_svm, precision_svm, color='m', alpha=0.2,
		         where='post')
		plt.step(recall_log, precision_log, color='r', alpha=0.2,
		         where='post')
		plt.step(recall_boost, precision_boost, color='b', alpha=0.2,
		         where='post')
		plt.step(recall_net2, precision_net2, color='y', alpha=0.2,
		         where='post')
		plt.step(recall_net3, precision_net3, color='g', alpha=0.2,
		         where='post')

		plt.fill_between(recall_svm, precision_svm, alpha=0.2, color='m', **step_kwargs)
		plt.fill_between(recall_log, precision_log, alpha=0.2, color='r', **step_kwargs)
		plt.fill_between(recall_boost, precision_boost, alpha=0.2, color='b', **step_kwargs)
		plt.fill_between(recall_net2, precision_net2, alpha=0.2, color='y', **step_kwargs)
		plt.fill_between(recall_net3, precision_net3, alpha=0.2, color='g', **step_kwargs)

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('2-class Precision-Recall Model Select {}, {}# Components'.format('LR_MODEL',i))
		plt.savefig('RECPRE_{}_{}.png'.format('LR_MODEL',i),dpi=300)
		plt.show()

		print("---------------------------------",file=f)
		print("CLASS REPORT {}:\n{}\n".format('DeepNet2',classification_report(y_test,np.round(y_net2,0))),file=f)
		print("AUC REPORT {}:\n{}\n".format('DeepNet2',roc_auc_score(y_test,y_net2)),file=f)
		print("TIME {} : {}\n".format('DeepNet2',time_dn2),file=f)
		print("---------------------------------",file=f)
		print("CLASS REPORT {}:\n{}\n".format('DeepNet3',classification_report(y_test,np.round(y_net3,0))),file=f)
		print("AUC REPORT {}:\n{}\n".format('DeepNet3',roc_auc_score(y_test,y_net3)),file=f)
		print("TIME {} : {}\n".format('DeepNet2',time_dn3),file=f)
		print("---------------------------------",file=f)
		print("CLASS REPORT {}:\n{}\n".format('Logistic Regression CW 2, L2',classification_report(y_test,y_log)),file=f)
		print("AUC REPORT {}:\n{}\n".format('Logistic Regression CW 2, L2',roc_auc_score(y_test,y_log)),file=f)
		print("TIME {} : {}\n".format('DeepNet2',time_log),file=f)
		print("---------------------------------",file=f)
		print("CLASS REPORT {}:\n{}\n".format('SVM',classification_report(y_test,y_svm)),file=f)
		print("AUC REPORT {}:\n{}\n".format('SVM',roc_auc_score(y_test,y_svm)),file=f)
		print("TIME {} : {}\n".format('DeepNet2',time_svm),file=f)
		print("---------------------------------",file=f)

