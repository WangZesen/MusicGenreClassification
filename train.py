import mxnet as mx
import dataLoader, deploy, symbol, debug

network = symbol.get_symbol(num_classes = 10)
net = mx.mod.Module(symbol = network, context = mx.gpu(0))

trainDataIter = dataLoader.get_train_data_iter()
valDataIter = dataLoader.get_val_data_iter()

num_epoch = 100

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
print results

results = net.score(trainDataIter, deploy.get_eval_metric(), reset = True)
print results

net.save_params('models/test_{}.params'.format(str(num_epoch)))

'''
num_epoch = 100,
epoch_size = 10,
optimizer = deploy.get_optimizer(),
initializer = deploy.get_initializer()
'''

