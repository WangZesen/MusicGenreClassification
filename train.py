import mxnet as mx
import dataLoader, deploy, symbol, debug


trainDataIter = dataLoader.get_train_data_iter()
valDataIter = dataLoader.get_val_data_iter()

network = symbol.get_symbol(num_classes = 10)

net = mx.mod.Module(symbol = network, context = mx.gpu(0))

num_epoch = 50

net.fit(train_data = trainDataIter,
		eval_data = valDataIter,
		epoch_end_callback = debug.epoch_end_callback,
		eval_end_callback = debug.eval_end_callback,
		eval_metric = ['MSE', 'accuracy'],
		optimizer = "sgd",
		initializer = deploy.get_initializer(),
		num_epoch = num_epoch
		)

results = net.score(valDataIter, ['MSE', 'accuracy'], reset = True)
print results

results = net.score(trainDataIter, ['MSE', 'accuracy'], reset = True)
print results

net.save_params('models/test_{}.params'.format(str(num_epoch)))

'''
num_epoch = 100,
epoch_size = 10,
optimizer = deploy.get_optimizer(),
initializer = deploy.get_initializer()
'''

