import tflearn as tf
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_csv
import tensorflow as tfs


data,target=load_csv('dataset.csv',target_column=-1,columns_to_ignore=[1],has_header=True,categorical_labels=True,n_classes=2)
sess=tfs.Session()
outp = tfs.gather(data,2)
print(sess.run(outp))
outp1 = tfs.gather(target,2)
print(sess.run(outp1))

network = input_data(shape=[None,23],name='input')
network = fully_connected(network,10,activation='relu',name='nn_layer_1')
network = fully_connected(network,5,activation='relu',name='nn_layer_2')
network = fully_connected(network,2,activation='softmax',name='output_layer')
network = regression(network,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
model = tf.DNN(network, tensorboard_verbose=3)
#model.fit(data,target,n_epoch=50,validation_set=0.3,show_metric=True,run_id='model1')
