import numpy as np
import mxnet as mx
import pickle


def getDataIter():
	dataFile = open("data/data", 'r')
	dataContent = dataFile.read()
	data = np.asarray(pickle.loads(dataContent))
	print data.shape

	labelFile = open("data/label", 'r')
	labelContent = labelFile.read()
	label = np.asarray(pickle.loads(labelContent))

	dataiter = mx.io.NDArrayIter(data, label, 3, True, last_batch_handle='discard')
	#for batch in dataiter:
	#	print batch.data[0].asnumpy()
	#	print batch.data[0].shape
	#	print batch.label
	return dataiter

