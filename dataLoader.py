import numpy as np
import mxnet as mx
import pickle


def get_val_data_iter():
	dataFile = open("data/valData", 'r')
	dataContent = dataFile.read()
	data = np.asarray(pickle.loads(dataContent))
	print data.shape

	labelFile = open("data/valLabel", 'r')
	labelContent = labelFile.read()
	label = np.asarray(pickle.loads(labelContent))

	dataiter = mx.io.NDArrayIter(data, label, 5, True, last_batch_handle='discard')
	return dataiter

def get_train_data_iter():
	dataFile = open("data/trainData", 'r')
	dataContent = dataFile.read()
	data = np.asarray(pickle.loads(dataContent))
	print data.shape

	labelFile = open("data/trainLabel", 'r')
	labelContent = labelFile.read()
	label = np.asarray(pickle.loads(labelContent))

	dataiter = mx.io.NDArrayIter(data, label, 5, True, last_batch_handle='discard')
	return dataiter

