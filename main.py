
import tensorflow as tf
import numpy as np
from model import Unet
import argparse
import os
from utils import get_data_set, batch_bgr2gray, build_log_dir
from tqdm import tqdm,trange
from Benchmarks import Benchmarks
import sys

def main():
    '''
    Args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
    parser.add_argument('--name', type=str, help='Name of experiment.')
    parser.add_argument('--batch-size', type=int, default=3, help='Mini-batch size.')
    parser.add_argument('--image-size', type=int, default=48, help='Size of random crops used for training samples.')
    parser.add_argument('--classes', type=int, default=1, help='classes number')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''
    PlaceHolder feed data
    '''
    x_train = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='x_train')
    y_true = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='y_true')

    print('__DEBUG__NAME', x_train)

    unet = Unet(batch_size=args.batch_size, classes=args.classes,img_size=args.image_size)
    y_pred = unet.create_unet(x_train)

    y_loss = unet.loss_function(y_true, y_pred)

    y_pred = tf.argmax(y_pred, axis = 3, name="y_pred")

    optimizer = unet.optimize(y_loss)


    train_path = './dataset/PreprocessedData.h5'
    # train_label = './dataset/..'

    train_data = get_data_set(train_path, 'train')
    train_label = get_data_set(train_path, 'label')

    # Create log folder
    if args.load and not args.name:
        log_path = os.path.dirname(args.load)
    else:
        log_path = build_log_dir(args, sys.argv)


    benchmarks = [
        Benchmarks('./Benchmarks/retina_test/', name='retina image')
    ]

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=100)

        iteration = 0
        n_iteration_per_epoch = len(train_data) // args.batch_size

        # Load all
        if args.load:
            iteration = int(args.load.split('-')[-1])
            saver.restore(sess, args.load)
            print(saver)
            print("load_process_DEBUG")


        while True:

            t =trange(0, len(train_data) - args.batch_size + 1, args.batch_size, desc='Iterations')
            total_loss = 0


            for batch_idx in t:

                #     # y_pred = unet.create_unet(x_train, train=False)

                #     for benchmark in benchmarks:
                #         benchmark.evaluate(sess, y_pred, log_path, iteration)


                batch_train = train_data[batch_idx:batch_idx + args.batch_size]
                batch_train = batch_bgr2gray(batch_train)
                batch_train = np.expand_dims(batch_train, axis=-1)
                batch_train = np.multiply(batch_train, 1.0 / 255.0)

                batch_label = train_label[batch_idx:batch_idx + args.batch_size]
                batch_label = batch_bgr2gray(batch_label)
                batch_label = np.expand_dims(batch_label, axis=-1)
                batch_label = np.multiply(batch_label, 1.0 / 255.0)
                # print('__DEBUG__', batch_label.shape)

                feed_dict_tr = {x_train: batch_train, y_true: batch_label}

                y_arr = sess.run(y_pred, feed_dict=feed_dict_tr)
                sess.run(optimizer, feed_dict=feed_dict_tr)
                loss = sess.run(y_loss, feed_dict=feed_dict_tr)

                total_loss += loss

                cont = str(np.max(y_arr)/100) + ": " + str(total_loss / n_iteration_per_epoch)

                t.set_description("%s" % cont)

                # _, err = sess.run([optimizer, y_loss],\
                         # feed_dict={x:batch_train,y_true:batch_label})


                iteration += 1
                if iteration%100 == 0:
                    saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=True)





if __name__ == '__main__':
	main()

























