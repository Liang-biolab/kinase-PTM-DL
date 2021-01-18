import tensorflow as tf
import tensorflow.keras.metrics as metrics
from keras import models
from keras import layers, Input, regularizers
from keras import optimizers
from keras import activations
from keras.utils import np_utils
from keras import backend as K
from tensorflow.keras import callbacks
import numpy as np 
import datetime
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def get_data_label(fea):
	if fea == 'all':
		fea = 'seq+str+dyn'

	data = np.load('data/' + fea + '/test_data_1d.npy', allow_pickle=True).astype(float)
	label = np.load('data/' + fea + '/test_label_1d.npy').astype(float)

	label = np_utils.to_categorical(label)

	return data, label


def predict(data,fea):
	if fea == 'all':
		fea = 'seq+str+dyn'

	model_num = 5
	pred_prob = None
	for i in range(model_num):
		model = load_model(fea + '/' + fea + '_model/fnn_3class_'+str(i)+'.h5', compile=False)
		if i == 0:
			pred_prob = model.predict(data)
		else:
			pred_prob += model.predict(data)

	pred_prob /= 5

	return pred_prob


if __name__ == '__main__':
	fea = 'all'

	data, label = get_data_label(fea)
	pred_prob = predict(data, fea)
	pred_label = np.argmax(pred_prob, axis=1)

	print(pred_label)


