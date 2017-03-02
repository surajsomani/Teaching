import tflearn
from tflearn.layers.core import fully_connected,input_data
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1,784])
test_x = test_x.reshape([-1,784])
NN = input_data(shape=[None,784],name='input')
layer_1 = fully_connected(NN,500,activation='relu')
layer_2 = fully_connected(layer_1,200,activation='relu')
layer_3 = fully_connected(layer_2,50,activation='relu')
output = fully_connected(layer_3,10, activation='softmax')
output = regression(output, optimizer='adam', learning_rate=0.01,name='target')
model = tflearn.DNN(output)
model.fit({'input':X},{'target':Y}, n_epoch=10, validation_set=({'input':test_x}, {'target':test_y}),snapshot_step=500, show_metric=True, run_id='mymodel', )


