import mxnet as mx
import dataLoader, deploy, symbol, debug


dataIter = dataLoader.get_train_data_iter()
network = symbol.get_symbol(num_classes = 10)

net = mx.mod.Module(symbol = network,
					context = mx.gpu(0))

net.fit(train_data = dataIter,
		eval_data = dataIter,
		epoch_end_callback = debug.epoch_end_callback,
		eval_end_callback = debug.eval_end_callback,
		eval_metric = 'MSE',
		optimizer = "sgd",
		initializer = deploy.get_initializer(),
		num_epoch = 100
		)

results = net.score(dataIter, ['MSE', 'accuracy'], reset = True)
print results

'''
num_epoch = 100,
epoch_size = 10,
optimizer = deploy.get_optimizer(),
initializer = deploy.get_initializer()
'''

