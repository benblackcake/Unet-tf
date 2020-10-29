

import tensorflow as tf
import numpy as np
import cv2
import scipy.misc
from utils import get_data_set, batch_bgr2gray, build_log_dir
from model import Unet
from tqdm import tqdm,trange
import argparse
from Benchmarks import Benchmarks
import os
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
    parser.add_argument('--name', type=str, help='Name of experiment.')
    parser.add_argument('--batch-size', type=int, default=3, help='Mini-batch size.')
    parser.add_argument('--image-size', type=int, default=48, help='Size of random crops used for training samples.')
    parser.add_argument('--classes', type=int, default=1, help='classes number')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    x_train = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='x_train')
    unet = Unet(batch_size=args.batch_size, classes=args.classes,img_size=args.image_size)
    y_pred = unet.create_unet(x_train, train=False)

    # Create log folder
    if args.load and not args.name:
        log_path = os.path.dirname(args.load)
    else:
        log_path = build_log_dir(args, sys.argv)


    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        benchmarks = [
            Benchmarks('./Benchmarks/retina_test/', name='retina image')
        ]
        # Load all
        saver = tf.train.Saver()

        if args.load:
            iteration = int(args.load.split('-')[-1])
            saver.restore(sess, args.load)
            print(saver)
            print("load_process_DEBUG")

        for benchmark in benchmarks:
            benchmark.evaluate(sess, y_pred, x_train, log_path, iteration)





















