import tensorflow as tf
import numpy as np


class Unet(object):

    def __init__(self, batch_size, classes, img_size):
        self.batch_size = batch_size
        self.classes = classes
        self.img_size = img_size
        

    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def _create_biases(self, size):
        return tf.Variable(tf.constant(0.05, shape=[size]))


    def _conv_layer(self, input, num_input_channels, conv_filter_size,\
                   num_filters, padding='SAME', relu=True):

        weights = self._create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = self._create_biases(num_filters)
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)
        layer += biases

        if relu:
            layer = tf.nn.relu(layer)
        return layer


    def _pool_layer(self, input, padding='SAME'):

        return tf.nn.max_pool(value=input,\
              ksize = [1, 2, 2, 1],\
              strides=[1, 2, 2, 1],\
              padding=padding)

    def _un_conv(self, input, num_input_channels, conv_filter_size,\
                num_filters, feature_map_size, train=True,\
                padding='SAME',relu=True):


        weights = self._create_weights(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
        biases = self._create_biases(num_filters)

        if train:
            batch_size_0 = self.batch_size
        else:
            batch_size_0 = 1

        layer = tf.nn.conv2d_transpose(value=input, filter=weights,\
                           output_shape=[batch_size_0, feature_map_size, feature_map_size, num_filters],\
                           strides=[1, 2, 2, 1],\
                           padding=padding)
        layer += biases

        if relu:
            layer = tf.nn.relu(layer)
            
        return layer


    def create_unet(self, x_train, train=True):

        # train is used for un_conv, to determine the batch size
        # with tf.variable_scope('forward') as scope:

        # train is used for un_conv, to determine the batch size

        conv1 = self._conv_layer(x_train, 1, 3, 64)
        conv2 = self._conv_layer(conv1, 64, 3, 64)
        pool2 = self._pool_layer(conv2)
        conv3 = self._conv_layer(pool2, 64, 3, 128)
        conv4 = self._conv_layer(conv3, 128, 3, 128)
        pool4 = self._pool_layer(conv4)
        conv5 = self._conv_layer(pool4, 128, 3, 256)
        conv6 = self._conv_layer(conv5, 256, 3, 256)
        pool6 = self._pool_layer(conv6)
        conv7 = self._conv_layer(pool6, 256, 3, 512)
        conv8 = self._conv_layer(conv7, 512, 3, 512)
        pool8 = self._pool_layer(conv8)

        conv9 = self._conv_layer(pool8, 512, 3, 1024)
        conv10 = self._conv_layer(conv9, 1024, 3, 1024)

        conv11 = self._un_conv(conv10, 1024, 2, 512, self.img_size // 8, train)
        merge11 = tf.concat(values=[conv8, conv11], axis = -1)

        conv12 = self._conv_layer(merge11, 1024, 3, 512)
        conv13 = self._conv_layer(conv12, 512, 3, 512)

        conv14 = self._un_conv(conv13, 512, 2, 256, self.img_size // 4, train)
        merge14 = tf.concat([conv6, conv14], axis=-1)

        conv15 = self._conv_layer(merge14, 512, 3, 256)
        conv16 = self._conv_layer(conv15, 256, 3, 256)

        conv17 = self._un_conv(conv16, 256, 2, 128, self.img_size // 2, train)
        merge17 = tf.concat([conv17, conv4], axis=-1)

        conv18 = self._conv_layer(merge17, 256, 3, 128)
        conv19 = self._conv_layer(conv18, 128, 3, 128)

        conv20 = self._un_conv(conv19, 128, 2, 64, self.img_size, train)
        merge20 = tf.concat([conv20, conv2], axis=-1)

        conv21 = self._conv_layer(merge20, 128, 3, 64)
        conv22 = self._conv_layer(conv21, 64, 3, 64)
        conv23 = self._conv_layer(conv22, 64, 1, self.classes)


        # logits = tf.nn.sigmoid(conv23)

        return conv23

    def loss_function(self, y_true, y_pred):

        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)

        return 1. - (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

    def optimize(self, loss): 
        # tf.control_dependencies([discrim_train
        # update_ops needs to be here for batch normalization to work
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='forward')
        # with tf.control_dependencies(update_ops):
        return tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # def loss_function(self, y_pred, y_true):
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        # cost = tf.reduce_mean(cross_entropy)
        # tf.add_to_collection("losses", cost)
        # cost_reg = tf.add_n(tf.get_collection("losses"))

        # tf.summary.scalar("loss", cost_reg)
        # return tf.compat.v1.keras.backend.binary_crossentropy(y_true, y_pred)















