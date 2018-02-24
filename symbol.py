import mxnet as mx

eps = 1e-10 + 1e-5
bn_mom = 0.9
fix_gamma = False


# def convolutionLayer

def get_symbol(num_classes = 10):
	data = mx.symbol.Variable('data')
	data_bn = mx.symbol.BatchNorm(data = data, fix_gamma = fix_gamma, eps = eps, name = "data_batch_norm")

	conv1 = mx.symbol.Convolution(data = data_bn, num_filter = 256, kernel = (4, 4), stride = (1, 1), pad = (2, 2), name = "conv1")
	conv1_bn = mx.symbol.BatchNorm(data = conv1, fix_gamma = fix_gamma, eps = eps, name = "conv1_batch_norm")
	conv1_act = mx.symbol.Activation(data= conv1_bn, act_type='relu', name='conv1_relu')
	pool1 = mx.symbol.Pooling(data = conv1_act, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max", name = "pool1")

	conv2 = mx.symbol.Convolution(data = pool1, num_filter = 128, kernel = (4, 4), stride = (1, 1), pad = (2, 2), name = "conv2")
	conv2_bn = mx.symbol.BatchNorm(data = conv2, fix_gamma = fix_gamma, eps = eps, name = "conv2_batch_norm")
	conv2_act = mx.symbol.Activation(data= conv2_bn, act_type='relu', name='conv2_relu')
	pool2 = mx.symbol.Pooling(data = conv2_act, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max", name = "pool2")

	conv3 = mx.symbol.Convolution(data = pool2, num_filter = 128, kernel = (4, 4), stride = (1, 1), pad = (2, 2), name = "conv3")
	conv3_bn = mx.symbol.BatchNorm(data = conv3, fix_gamma = fix_gamma, eps = eps, name = "conv3_batch_norm")
	conv3_act = mx.symbol.Activation(data= conv3_bn, act_type='relu', name='conv3_relu')
	pool3 = mx.symbol.Pooling(data = conv3_act, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max", name = "pool3")

	conv4 = mx.symbol.Convolution(data = pool3, num_filter = 64, kernel = (2, 2), stride = (1, 1), pad = (1, 1), name = "conv4")
	conv4_bn = mx.symbol.BatchNorm(data = conv4, fix_gamma = fix_gamma, eps = eps, name = "conv4_batch_norm")
	conv4_act = mx.symbol.Activation(data= conv4_bn, act_type='relu', name='conv4_relu')
	pool4 = mx.symbol.Pooling(data = conv4_act, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max", name = "pool4")

	conv5 = mx.symbol.Convolution(data = pool4, num_filter = 64, kernel = (2, 2), stride = (1, 1), pad = (1, 1), name = "conv5")
	conv5_bn = mx.symbol.BatchNorm(data = conv5, fix_gamma = fix_gamma, eps = eps, name = "conv5_batch_norm")
	conv5_act = mx.symbol.Activation(data= conv5_bn, act_type='relu', name='conv5_relu')
	pool5 = mx.symbol.Pooling(data = conv5_act, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max", name = "pool5")

	conv6 = mx.symbol.Convolution(data = pool5, num_filter = 64, kernel = (2, 2), stride = (1, 1), pad = (1, 1), name = "conv6")
	conv6_bn = mx.symbol.BatchNorm(data = conv6, fix_gamma = fix_gamma, eps = eps, name = "conv6_batch_norm")
	conv6_act = mx.symbol.Activation(data= conv6_bn, act_type='relu', name='conv6_relu')
	pool6 = mx.symbol.Pooling(data = conv6_act, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max", name = "pool6")

	conv7 = mx.symbol.Convolution(data = pool6, num_filter = 64, kernel = (2, 2), stride = (1, 1), pad = (1, 1), name = "conv7")
	conv7_bn = mx.symbol.BatchNorm(data = conv7, fix_gamma = fix_gamma, eps = eps, name = "conv7_batch_norm")
	conv7_act = mx.symbol.Activation(data= conv7_bn, act_type='relu', name='conv7_relu')
	pool7 = mx.symbol.Pooling(data = conv7_act, kernel=(2, 2), stride=(2, 2), pad=(0, 0), pool_type="max", name = "pool7")

	pool5_flat = mx.symbol.Flatten(data = pool5, name = "pool5_flatten")
	pool6_flat = mx.symbol.Flatten(data = pool6, name = "pool6_flatten")
	pool7_flat = mx.symbol.Flatten(data = pool7, name = "pool7_flatten")

	concate = mx.symbol.Concat(pool5_flat, pool6_flat, pool7_flat)

	fc1 = mx.symbol.FullyConnected(data = concate, num_hidden = num_classes)

	softmax = mx.symbol.SoftmaxOutput(data = fc1, name='softmax')

	'''
	arg_shape, output_shape, aux_shape = concate.infer_shape(data = (3, 5, 599, 128))
	print arg_shape
	print output_shape
	print aux_shape
	'''

	return softmax


if __name__ == "__main__":
	get_symbol()