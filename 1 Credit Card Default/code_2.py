import pandas as pd 
import tensorflow as tf
import numpy as np
df=pd.read_csv('dataset.csv',usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],skiprows = [0],header=1)
d = df.values
l = pd.read_csv('dataset.csv',usecols = [24] ,header=1)
labels = l.values
data = np.float32(d)
labels = np.array(l)
print(data)
# x = tf.placeholder(tf.float32,shape=(150,5))
# x = data
# w = tf.random_normal([100,150],mean=0.0, stddev=1.0, dtype=tf.float32)
# y = tf.nn.softmax(tf.matmul(w,x))

# with tf.Session() as sess:
#     print sess.run(y)
trainX = tf.pack(data)
trainY = tf.pack(labels)

x = tf.placeholder('float', [None, 23])
y = tf.placeholder('float')

hm_epochs= 100
n_classes=2
batch_size=100
n_nodes_hl1 = 10

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([23, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']
    
    return output
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict={x: data, y: labels})
            epoch_loss += c
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:data, y:labels}))
train_neural_network(x)

	