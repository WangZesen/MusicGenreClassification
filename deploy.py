import mxnet as mx
import os


def get_optimizer():
	return mx.optimizer.SGD(momentum = 0.9)

def get_begin_epoch():
	maxEpoch = 0;
	fileList = os.listdir('models')
	for fileName in fileList:
		if int(fileName.split('_')[1].split('.')[0]) > maxEpoch:
			maxEpoch = int(fileName.split('_')[1].split('.')[0])
	return maxEpoch

def get_initializer():
	maxEpoch = -1;
	fileList = os.listdir('models')
	targetFile = None;
	for fileName in fileList:
		if int(fileName.split('_')[1].split('.')[0]) > maxEpoch:
			maxEpoch = int(fileName.split('_')[1].split('.')[0])
			targetFile = fileName
	if maxEpoch == -1:
		return mx.initializer.Normal(sigma = 0.01)
	else:
		return mx.initializer.Load('models/' + targetFile)

def get_eval_metric():
	top_k = 3
	eval_metric = mx.metric.CompositeEvalMetric()
	eval_metric.add(mx.metric.Accuracy())
	eval_metric.add(mx.metric.TopKAccuracy(top_k = 3, name = "top_{}_accuracy".format(str(top_k))))
	return eval_metric
	
if __name__ == "__main__":
	x = get_optimizer()
