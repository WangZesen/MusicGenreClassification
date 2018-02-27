import mxnet as mx
import numpy as np
import pickle

class DiskDataIter(mx.io.DataIter):
	def __init__(self, file_list = None, label_list = None, batch_num = 1):
		
		assert not (file_list == None)
		assert not (label_list == None)
		assert batch_num <= len(file_list)
		
		data_names = ["data"]
		label_names = ["softmax_label"]
		data_shapes = [(batch_num,) + self.get_shape(file_list[0])]
		label_shapes = [(batch_num,)]
		
		self._provide_data = list(zip(data_names, data_shapes))
		self._provide_label = list(zip(label_names, label_shapes))
		self.num_batches = len(file_list)
		self.cur_batch = 0
		self.batch_num = batch_num
		self.file_list = file_list
		self.label_list = label_list
	
	def get_shape(self, file_name):
		f = open(file_name, "r")
		content = f.read()
		sample = np.asarray(pickle.loads(content))
		return sample.shape
	
	def __iter__(self):
		return self

	def reset(self):
		self.cur_batch = 0

	def __next__(self):
		return self.next()

	@property
	def provide_data(self):
		return self._provide_data

	@property
	def provide_label(self):
		return self._provide_label

	def next(self):
		if self.cur_batch + self.batch_num <= self.num_batches:
			data = []
			label = []
			for i in range(self.batch_num):
				f = open(self.file_list[self.cur_batch + i], "r")
				content = f.read()
				cur_data = np.asarray(pickle.loads(content))
				data.append(cur_data)
				label.append(self.label_list[self.cur_batch + i])
			data = [mx.nd.array(np.array(data))]
			label = [mx.nd.array(label)]
			self.cur_batch += self.batch_num
			
			# data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
			# label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
			
			return mx.io.DataBatch(data, label)
			
		else:
			raise StopIteration

if __name__ == "__main__":
	test = DiskDataIter(["extractData/blues.00000.pp", "extractData/blues.00001.pp"], [0, 0], 1)
	print test.provide_label
	print test.provide_data
	print test.next()
	
	
