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
        conv1 = tf.layers.conv2d(inputs=x_train, filters=16, kernel_size=3, strides=1,
                         padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_uniform())
        conv1 = tf.layers.dropout(inputs=conv1, rate=0.1)
        conv1 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding="valid")

        conv2 = tf.layers.conv2d(inputs=p1, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv2 = tf.layers.dropout(inputs=conv2, rate=0.1)
        conv2 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding="valid")

        conv3 = tf.layers.conv2d(inputs=p2, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv3 = tf.layers.dropout(inputs=conv3, rate=0.1)
        conv3 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding="valid")

        conv4 = tf.layers.conv2d(inputs=p3, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv4 = tf.layers.dropout(inputs=conv4, rate=0.1)
        conv4 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        p4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2, padding="valid")

        conv5 = tf.layers.conv2d(inputs=p4, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv5 = tf.layers.dropout(inputs=conv5, rate=0.2)
        conv5 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())

        up1 = tf.layers.conv2d_transpose(inputs=conv5, filters=128, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up1 = tf.concat([up1, conv4], axis=3)
        conv6 = tf.layers.conv2d(inputs=up1, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv6 = tf.layers.dropout(inputs=conv6, rate=0.2)
        conv6 = tf.layers.conv2d(inputs=conv6, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())

        up2 = tf.layers.conv2d_transpose(inputs=conv6, filters=64, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up2 = tf.concat([up2, conv3], axis=3)
        conv7 = tf.layers.conv2d(inputs=up2, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv7 = tf.layers.dropout(inputs=conv7, rate=0.2)
        conv7 = tf.layers.conv2d(inputs=conv7, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())

        up3 = tf.layers.conv2d_transpose(inputs=conv7, filters=32, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up3 = tf.concat([up3, conv2], axis=3)
        conv8 = tf.layers.conv2d(inputs=up3, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv8 = tf.layers.dropout(inputs=conv8, rate=0.2)
        conv8 = tf.layers.conv2d(inputs=conv8, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())

        up4 = tf.layers.conv2d_transpose(inputs=conv8, filters=16, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.initializers.glorot_uniform())
        up4 = tf.concat([up4, conv1], axis=3)
        conv9 = tf.layers.conv2d(inputs=up4, filters=16, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())
        conv9 = tf.layers.dropout(inputs=conv9, rate=0.1)
        conv9 = tf.layers.conv2d(inputs=conv9, filters=16, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.glorot_uniform())

        conv10 = tf.layers.conv2d(inputs=conv9, filters=1, kernel_size=1, strides=1,
                                  kernel_initializer=tf.initializers.glorot_uniform())
        logits = tf.nn.sigmoid(conv10)

        # logits = tf.nn.sigmoid(conv23)

        return logits

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















