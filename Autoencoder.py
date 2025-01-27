import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense, LeakyReLU
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, optimizers
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NearMiss

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def main():
	from pandas import read_csv
	# Read in the data
	DATA = read_csv("norm_data__non_log.txt",sep='\t').T
	DATA = DATA.apply(np.log).values # Retain the log 10 due to the maximising values
	label = read_csv("sample_list.csv",sep=';')

	# Conversion of string to bool
	mapping = {'Non-LCa':0,'LCa':1}
	TARGET = label.Disease.map(mapping).values

	

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(DATA,TARGET,test_size=0.5,shuffle=True)


	ada = ADASYN(sampling_strategy='minority')
	X_train, y_train = ada.fit_resample(X_train,y_train)

	# 	# normalise?
	# scaler = StandardScaler().fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_test = scaler.transform(X_test)

	nb_epoch = 1000
	batch_size = 150
	input_dim = DATA.shape[1] 
	learning_rate = 1e-3
	decay = learning_rate/nb_epoch


	input_layer = Input(shape=(input_dim, ))

	encoder = Dense(2,activation='relu',use_bias=True)(input_layer)

	output_layer = Dense(input_dim, activation= 'relu',use_bias=True)(encoder)

	
	adam = optimizers.Adam(lr=learning_rate,decay=decay)
	autoencoder = Model(inputs=input_layer, outputs=output_layer)
	autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=adam)

	encoder_ = Model(input_layer, encoder)

	cp = ModelCheckpoint(filepath="autoencoder_lca_new.h5",
	                               save_best_only=True,
	                               verbose=0)

	tb = TensorBoard(log_dir='./logs',
	                histogram_freq=0,
	                write_graph=True,
	                write_images=True)

	history = autoencoder.fit(X_train, X_train,
	                    epochs=nb_epoch,
	                    batch_size=batch_size,
	                    shuffle=True,
	                    validation_data=(X_test, X_test),
	                    verbose=1,
	                    callbacks=[cp, tb]).history

	X_new = encoder_.predict(X_test)

	# print(X_new)
	# print('F1 {}'.format(f1_score(np.round(X_new,0),y_test)))

	from matplotlib import pyplot
	from mpl_toolkits.mplot3d import Axes3D
	import random


	# fig = pyplot.figure()
	# ax = Axes3D(fig)

	fig, ax = plt.subplots()

	ax.scatter(X_new[:,0],X_new[:,1],c=y_test)
	plt.show()


	plt.plot(history['loss'], linewidth=2, label='Train')
	plt.plot(history['val_loss'], linewidth=2, label='Test')
	plt.legend(loc='upper right')
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	#plt.ylim(ymin=0.70,ymax=1)
	plt.show()



if __name__ == '__main__':
	main()






