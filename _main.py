import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import time 

# set mode
caffe.set_device(0)
caffe.set_mode_gpu()

# load solver
solver = caffe.get_solver('./lenet_solver.prototxt')

# initialize using the trained weights
shapes = [(v[0].data.shape, v[1].data.shape) for v in solver.net.params.values()]
keys = solver.net.params.keys()
for l in range(4):  # each layer
	for i in range(2):  # each layer has two values: weights and biases
		solver.net.params[keys[l]][i].data[...] = np.reshape(
            np.loadtxt('../trained_params/param_{0}.txt'.format(2*l+i)), shapes[l][i])

# the last two layers' weights are wrong, should be the transpose.
solver.net.params[keys[2]][0].data[...] = np.reshape( np.loadtxt('../trained_params/param_4.txt'), (800,500)).transpose()
solver.net.params[keys[3]][0].data[...] = np.reshape( np.loadtxt('../trained_params/param_6.txt'), (500,10)).transpose()

solver.net.forward()  # train net: the loss is different.
solver.test_nets[0].forward() # test net: it did not give us a good result!!!
