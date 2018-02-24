import mxnet as mx

def epoch_end_callback(epoch, symbol, arg_params, aux_params):
	if (epoch + 1) % 10 == 0:
		print "[Train] Finished", epoch + 1, "iterations"

def eval_end_callback(info):
	# print info
	if (info.epoch + 1) % 2 == 0:
		metric, values = info.eval_metric.get()
		for i in range(len(metric)):
			print "[Validation Epoch {}]".format(str(info.epoch + 1)), metric[i] + ":", values[i]
