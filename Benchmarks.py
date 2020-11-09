
import numpy as np
from scipy import misc
from skimage.color import rgb2ycbcr, rgb2yuv
import cv2
import tensorflow as tf
import glob
import os
from utils import modcrop
import argparse

class Benchmarks():

    def __init__(self,path_train, name):
        self.name = name
        # self.images_lr, self.names = self.load_images_by_model(model='LR')
        # self.images_segment, self.names = self.load_images_by_model(path_train, model=args.test_filename)
        self.images_segment, self.names = self.load_images_by_model(path_train, model='test')
        # self.images_label, self.names = self.load_images_by_model(path_label, model='GT')

    def load_images_by_model(self, path, model, file_format='*'):
        """Loads all images that match '*_{model}.{file_format}' and returns sorted list of filenames and names"""
        # Get files that match the pattern
        filenames = glob.glob(os.path.join(path, '*_' + model + '.' + file_format))
        print('__DEBUG__fileName %s' % filenames)
        # Extract name/prefix eg: '/.../baby_LR.png' -> 'baby'
        names = [os.path.basename(x).split('_')[0] for x in filenames]
        return self.load_images(filenames), names


    def load_images(self, images):
        """Given a list of file names, return a list of images"""
        out = []
        for image in images:
            img = cv2.imread(image)
            out.append(modcrop(img, scale=4))
            # out.append(img)

        return out


    def save_image(self, image, path):
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)    
        misc.toimage(image, cmin=0, cmax=255).save(path)


    def save_images(self, images, log_path, iteration):
        count = 0
        # for output, lr, hr, name in zip(images, self.images_rain, self.images_label, self.names):
        for output, name in zip(images, self.names):
            # Save output
            path = os.path.join(log_path, self.name, name, '%d_out.png' % iteration)
            print('path:',path)
            print('debug_output',output.shape)
            # cv2.imshow('__DEBUG__image', output)
            # cv2.waitKey(0)
            self.save_image(output, path)

    def evaluate(self, sess, y_pred, log_path, iteration=0):

        pred = []

        for i, segment_img in enumerate(self.images_segment):

            # cv2.imshow('__DEBUG__image', segment_img)
            # cv2.waitKey(0)
            print('__DEBUG__SHAPE', segment_img.shape)
            h,w, _ = segment_img.shape
            segment_img = cv2.resize(segment_img,(96,96))
            # segment_img = np.expand_dims(segment_img, axis=-1)
            # print('__DEBUG__SHAPE', segment_img[np.newaxis].shape)
            print('SHAPE', segment_img.shape)

            pred_segment = sess.run(y_pred, feed_dict={"x_train:0":segment_img[np.newaxis]/255.})

            segment_result = np.squeeze(pred_segment, axis=0)
            segment_result *= 255
            segment_result = segment_result.astype('uint8')
            print('segment_result:',segment_result)
            segment_result = cv2.resize(segment_result,(w,h))
            print(segment_result.shape)
            pred.append(segment_result)

        self.save_images(pred, log_path, iteration)















