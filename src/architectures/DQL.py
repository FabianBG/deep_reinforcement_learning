import tensorflow as tf

class DQLValue():

    def __init__(self):
        self.name = "Arquitectura DQL input => Conv2d => Conv2d => Conv2d => Desde => Output"


    def fit(self, input, output_size):

        layer1 = tf.layers.conv2d(
            input,
            filters=32,
            kernel_size=[8,8],
            strides=[4,4],
            activation=tf.nn.elu,
            padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="layer1")
        
        layer2 = tf.layers.conv2d(
            layer1,
            filters=64,
            kernel_size=[4,4],
            strides=[2,2],
            activation=tf.nn.elu,
            padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="layer2")
        
        layer3 = tf.layers.conv2d(
            layer2,
            filters=128,
            kernel_size=[3,2],
            strides=[1,1],
            activation=tf.nn.elu,
            padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="layer3")

        layer4 = tf.layers.dense(layer3, 
            units=512,
            name="layer4",
            activation=tf.nn.elu)

        output = tf.layers.dense(layer4, 
            units=output_size,
            name="output",
            activation=None)
        
        return output