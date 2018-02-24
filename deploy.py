import mxnet as mx

def get_optimizer():
	return mx.optimizer.SGD(momentum = 0.9)

def get_initializer():
	return mx.initializer.Normal(sigma = 0.01)

if __name__ == "__main__":
	x = get_optimizer()
