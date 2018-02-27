import mxnet as mx
import dataLoader, deploy, symbol, debug

network = symbol.get_symbol(num_classes = 10)
net = mx.mod.Module(symbol = network, context = mx.gpu(0))

trainDataIter, valDataIter = dataLoader.get_data_iter()

num_epoch = 100

try:
	net.fit(train_data = trainDataIter,
			eval_data = valDataIter,
			epoch_end_callback = debug.epoch_end_callback,
			eval_end_callback = debug.eval_end_callback,
			eval_metric = deploy.get_eval_metric(),
			optimizer = "sgd",
			initializer = deploy.get_initializer(),
			num_epoch = num_epoch,
			begin_epoch = deploy.get_begin_epoch()
			)
			
	results = net.score(valDataIter, deploy.get_eval_metric(), reset = True)
	print "[Info] Validation Result:", results
	results = net.score(trainDataIter, deploy.get_eval_metric(), reset = True)
	print "[Info] Training Result:", results
	
	print "[Info] Saving Parameters..."
	net.save_params('models/test_{}.params'.format(str(num_epoch)))
	print "[Info] Finished"
			
except KeyboardInterrupt:
	print 
	print "[Info] Early Interrupt at", debug.recorder.epoch, "Iteration"
	print "[Info] Saving Parameters..."
	net.save_params('models/test_{}.params'.format(str(debug.recorder.epoch)))
	print "[Info] Finished"


