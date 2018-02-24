import mxnet as mx

def epoch_end_callback(epoch, symbol, arg_params, aux_params):
	if epoch % 10 == 0:
		print "finished", epoch, "iteration"

def eval_end_callback(info):
	if info.epoch % 10 == 0:
		print "eval", info.epoch
