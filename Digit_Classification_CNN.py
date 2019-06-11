# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:04:02 2019

@author: MUJ
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot = True)
def init_weight(shape):
    random_weight = tf.truncated_normal(shape,stddev =0.1)
    return tf.Variable(random_weight)
def init_bias(Shape):
    bias_value = tf.constant(0.1,shape = Shape)
    return tf.Variable(bias_value)
def conv_2d(x,w):
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')
def pool_2d2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
def convolutional_layer(image_x,shape):
    w = init_weight(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv_2d(image_x,w)+b)
def normal_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    w = init_weight([input_size,size])
    b = init_bias([size])
    return (tf.matmul(input_layer,w)+b)
x = tf.placeholder(tf.float32,shape = [None,784])
y = tf.placeholder(tf.float32,shape = [None,10])
image_x = tf.reshape(x,[-1,28,28,1])
conv_1 = convolutional_layer(image_x,[5,5,1,32])
conv_1_pooling = pool_2d2(conv_1)
conv_2 = convolutional_layer(conv_1_pooling,[5,5,32,64])
conv_2_pooling = pool_2d2(conv_2)
convo_2_flat = tf.reshape(conv_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_layer(convo_2_flat,1024))
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob = hold_prob)
y_pred = normal_layer(full_one_dropout,10)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate = .01)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        batch_x,batch_y = mnist.train.next_batch(50)
        feed = {x:batch_x,y:batch_y,hold_prob:0.5}
        sess.run(train,feed_dict = feed)
        if i%100 == 0:
            print("at every {} step".format(i))
            print("Accuracy")
            matches = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict = {x:mnist.test.images,y:mnist.test.labels,hold_prob:1.0}))
                    
        

 