import numpy as np
import mxnet as mx
import pickle, os, deploy, random, dataIter

def get_val_data_iter():
	dataFile = open("data/valData", 'r')
	dataContent = dataFile.read()
	data = np.asarray(pickle.loads(dataContent))
	print data.shape

	labelFile = open("data/valLabel", 'r')
	labelContent = labelFile.read()
	label = np.asarray(pickle.loads(labelContent))

	dataiter = mx.io.NDArrayIter(data, label, deploy.get_batch_size(), True, last_batch_handle='discard')
	return dataiter

def get_train_data_iter():
	dataFile = open("data/trainData", 'r')
	dataContent = dataFile.read()
	data = np.asarray(pickle.loads(dataContent))
	print data.shape

	labelFile = open("data/trainLabel", 'r')
	labelContent = labelFile.read()
	label = np.asarray(pickle.loads(labelContent))

	dataiter = mx.io.NDArrayIter(data, label, deploy.get_batch_size(), True, last_batch_handle='discard')
	return dataiter

def get_data_iter():
	numOfGenres = 10
	train_percent = 0.9
	genres = ["blues", "classic", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
	file_name_list = []
	label_list = []
	file_list = os.listdir("extractData")
	for file_name in file_list:
		if file_name.endswith(".pp"):
			file_name_list.append("extractData/" + file_name)
			for i in range(numOfGenres):
				if file_name.startswith(genres[i]):
					label_list.append(i)
					break
	concatenated = zip(file_name_list, label_list)
	random.shuffle(concatenated)
	file_name_list, label_list = zip(*concatenated)
	total_num = len(label_list)
	train_data = file_name_list[0: int(total_num * train_percent)]
	train_label = label_list[0: int(total_num * train_percent)]
	val_data = file_name_list[int(total_num * train_percent): total_num]
	val_label = label_list[int(total_num * train_percent): total_num]
	train_data_iter = dataIter.DiskDataIter(train_data, train_label, deploy.get_batch_size())
	val_data_iter = dataIter.DiskDataIter(val_data, val_label, deploy.get_batch_size())	
	return train_data_iter, val_data_iter
	
def get_data_iter_auto_encoder():
	train_percent = 0.9
	file_name_list = []
	label_list = []
	file_list = os.listdir("extractData")
	for file_name in file_list:
		if file_name.endswith(".pp"):
			file_name_list.append("extractData/" + file_name)
			label_list.append(0)
	concatenated = zip(file_name_list, label_list)
	random.shuffle(concatenated)
	file_name_list, label_list = zip(*concatenated)
	total_num = len(label_list)
	train_data = file_name_list[0: int(total_num * train_percent)]
	train_label = label_list[0: int(total_num * train_percent)]
	val_data = file_name_list[int(total_num * train_percent): total_num]
	val_label = label_list[int(total_num * train_percent): total_num]
	train_data_iter = dataIter.DiskDataIterEncoder(train_data, train_label, 5)
	val_data_iter = dataIter.DiskDataIterEncoder(val_data, val_label, 5)	
	return train_data_iter, val_data_iter

